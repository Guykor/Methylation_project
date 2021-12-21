import copy
import logging

import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from functools import partialmethod
import plots
from functions import build_profile, transform_distribution_df
from plots import admixture_results
from project_logger import get_logger

logger = get_logger(__name__)

# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


class AdmixtureEvaluator:
    def __init__(self, blood_samples, test_atlas, sim_config, wbc_dist=None):
        """
        :param blood_samples: methylation profiles of whole blood samples
        :param test_atlas: unseen_tissue_data
        """
        self.test = test_atlas
        self.rng = np.random.default_rng(seed=200793)
        self.sample_coverage = sim_config.get('required_coverage')
        self.subset_problem = sim_config.get('subset_problem')
        self.sampling_method = sim_config['sampling_method']
        self.wbc_name = sim_config.get('wbc_name')
        # an estimate used to sample reads by multiomial distribution.
        self.wbc_dist = wbc_dist
        self.sample_reads = self._set_sampling(self.sampling_method)

        self.alphas = sim_config.get('alphas')
        self.alpha_labels = sim_config.get('labels')
        self.bloods = blood_samples[: sim_config.get('num_blood_profiles')]
        self.num_profiles_rep = sim_config.get('num_profile_rep')
        self.debug = sim_config.get('debug')
        if self.bloods[0].index.equals(self.test.index):
            logger.warning("Blood samples and test data are not indexed the same.")

    def _set_sampling(self, method):
        if method == 'multinom':
            if self.wbc_dist is None:
                raise ValueError("for multiom sampling provide wbc_dist argument in ctor.")
            return self.sample_reads_multinom
        elif method == 'uniform':
            return self.sample_reads_uniform
        else:
            raise ValueError("sample_method should be 'multinom' or 'uniform'.")

    def sub_problem(self, model, cell_type):
        """
        Subset markers based on hard markers selection, needs update if using weighted selection.
        Blood samples follows the same index of cell profile by default.
        """
        subset = [cell_type, self.wbc_name]
        if self.subset_problem == 'subset_markers':
            # Not realistic since we can subset only our initial data markers (for the set of K cell types)
            # NO significant difference suppose to be compared to "all_markers" since given cell-types in our train
            # data, we already subset the marker in the model.
            marker_selection = model.V[model.V[subset].sum(axis=1) != 0].index.values
            sub_model = model.subset_data(subset, markers=marker_selection)
            cell_profile = self.test.get_tissue(cell_type, 'cnts').loc[marker_selection]
        elif self.subset_problem == 'all_markers':
            # Estimation is K=2. relaxation
            sub_model = model.subset_data(subset)
            cell_profile = self.test.get_tissue(cell_type, 'cnts')
        else:
            # do not subset
            sub_model = model
            cell_profile = self.test.get_tissue(cell_type, 'cnts')
        return sub_model, cell_profile

    def evaluate(self, model):
        """
        Runs evaluation process - simulate methylation profiles with mixed proportion of tissue reads, and record
        estimation of the model for the tissue mixed.
        :param model: modle with estimate_theta method.
        :return: dict: tissue_mixed -> dict: alpha -> predictions
        """
        result = {}
        full_model = model
        for cell_type in self.test.tissues:
            model, cell_profile = self.sub_problem(full_model, cell_type)
            print("Simulation of ", cell_type)
            start = time.time()
            predictions = self.evaluate_tissue(model, cell_type, cell_profile)
            print(cell_type, " evaluation took ", (time.time() - start) / 60, " minutes.")
            result[cell_type] = pd.DataFrame(predictions)

        plot_label = 'Admixture: ' + self.sampling_method + " sampling method"
        admixture_results(result, plot_label, self.alphas, self.alpha_labels)
        return result

    def evaluate_tissue(self, model, cell_type, cell_profile):
        """
        for the given cell_type, simulates for each alpha and each blood_sample, num_sim methylation profiles.
        then, the given model is used to estimate theta, and the theta[cell_type] is being recorded.
        :return: dict: alpha -> list of predicted theta[cell_type] results.
        """
        result = {}
        for alpha in tqdm(self.alphas, desc='alpha', ncols=80, ascii=True, leave=False):
            result[alpha] = []
            for i, blood_profile in tqdm(enumerate(self.bloods), desc="Bloods", ncols=80, ascii=True, leave=False):
                simulated = self.simulate_by_coverage(cell_type, blood_profile.loc[cell_profile.index],
                                                      cell_profile, alpha)
                if self.debug:
                    plots.mix_profile_check(simulated[:1], model.V, cell_type, alpha)

                for j, sim in tqdm(enumerate(simulated), desc='Simulated_dfs', ncols=80, ascii=True, leave=False):
                    predicted = model.estimate_theta(sim, cell_type + f" alpha: {alpha}, blood: {i}, sim: {j}")
                    result[alpha].append(predicted.loc[cell_type].item())
        return result

    def simulate_by_coverage(self, cell_type, blood_profile, cell_profile, alpha):
        """
        Coverage is defined by the average number of reads (observations) in a profile.
        The method simulates profile with required_coverage, by choosing randomly observations from the cell profile
        and the blood sample profile. s.t there will be alpha percent cell observations in the resulted profile.
        :param blood_profile: profile of shape N_markers, 4 (Pu, Pm, Px, N)
        :param cell_dist: distribution profile of shape N_markers, 4 (Pu, Pm, Px, N)
        :param alpha: float
        :return: list of self.num_sim simulated profiles
        """
        cell_coverage = cell_profile.N.mean()
        blood_coverage = blood_profile.N.mean()

        # what percent of reads do we take from the cell/blood profile? we than sample from each window that percentage.
        cell_mix_rate = alpha * (self.sample_coverage / cell_coverage)
        blood_mix_rate = (1 - alpha) * (self.sample_coverage / blood_coverage)

        cell_reads = (cell_profile.N * cell_mix_rate).round()
        blood_reads = (blood_profile.N * blood_mix_rate).round()

        diff_from_mean = round(alpha - np.mean(cell_reads / (cell_reads + blood_reads)), 4)
        diff_from_abs = round(alpha - (np.sum(cell_reads) / np.sum(cell_reads + blood_reads)), 4)
        logger.info(f"simulation cell_type diff of mean from alpha: {diff_from_mean}, "
                       f"diff from reads ratio is {diff_from_abs}")

        simulated = self.sample_reads(cell_reads, cell_profile, blood_reads, blood_profile)

        if self.debug:
            plots.mix_theta_check(alpha, cell_type, cell_reads, blood_reads)
        return [build_profile(profile, cell_profile.index) for profile in simulated]

    def sample_reads_multinom(self, cell_reads, cell_profile, blood_reads, blood_profile):
        """
        Samples required reads from every marker (row in table), by uniform selection of reads.
        :param cell_reads: vector of required reads per marker. if 0, zeros array will be returned.
        :param cell_dist: distribution profile
        :param blood_reads: vector of required reads per marker
        :return: array of shape [self.num_sim, N_markers, 3], for every i in range(Self.num_sim) is a simulated mixed
        profile.
        """
        cell_dist = transform_distribution_df(cell_profile)

        markers_indices = cell_reads.index.values
        samples = []
        for i in markers_indices:
            cell = self.rng.multinomial(cell_reads[i], cell_dist.loc[i][['U', 'M', 'X']], size=self.num_profiles_rep)
            blood = self.rng.multinomial(blood_reads[i], self.wbc_dist.loc[i][['U', 'M', 'X']],
                                         size=self.num_profiles_rep)
            samples.append(cell + blood)
        return np.swapaxes(np.array(samples), 0, 1)

    def sample_reads_uniform(self, cell_reads, cell_profile, blood_reads, blood_profile):
        """
        Samples required reads from every marker (row in table), by multinomial distribution defined in _dist dfs.
        :param cell_reads: vector of required reads per marker. if 0, zeros array will be returned.
        :param cell_dist: distribution profile
        :param blood_reads: vector of required reads per marker
        :param blood_dist: distribution profile
        :return: array of shape [self.num_sim, N_markers, 3], for every i in range(Self.num_sim) is a simulated mixed
        profile.
        """
        markers_indices = cell_reads.index.values
        samples = []
        for i in markers_indices:
            cell = self.sample_row_uniform(cell_profile.loc[i], cell_reads[i])
            blood = self.sample_row_uniform(blood_profile.loc[i], blood_reads[i])
            samples.append(cell + blood)
        return np.swapaxes(np.array(samples), 0, 1)

    def sample_row_uniform(self, row, req_reads):
        """
        Uniform choice of observations per row (marker).
        :param row: profile at a specific marker, contains UMX counts.
        :param req_reads: required reads overall in every simulated choice. if 0, zeros array of num_sim x 3 will be
        returned.
        :return: array of num_sim x 3 of the simulated data for that row.
        """
        read_list = np.array(['U'] * row.U + ['M'] * row.M + ['X'] * row.X)
        shuffled_indices = np.argsort(np.random.rand(self.num_profiles_rep, row.U + row.M + row.X), axis=1)
        sample_idx = shuffled_indices[:, :int(req_reads)]
        sample = read_list[sample_idx]
        result = np.array([(sample == 'U').sum(axis=1), (sample == 'M').sum(axis=1), (sample == 'X').sum(axis=1)]).T
        return result
