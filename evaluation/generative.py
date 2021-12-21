import numpy as np
import pandas as pd

from functions import build_profile
import plots

MULTINOM_DIM = 3


class GenerativeEvaluator:
    """
    Multi class Evaluation based on generative simulation of profiles.
    """

    def __init__(self, test_atlas, config):
        self.test = test_atlas

        self.rng = np.random.default_rng()
        self.num_prior_draws = config.get('num_prior_draws')
        self.num_sim = config.get('num_simulations')
        self.depth_args = config.get('depth_mean'), config.get('depth_std')

        self.sim_type = config['simulation']
        # How to preform the latent variables selection (if latent simulation type is selected).
        self.sample_latent = config.get('sample_latent')
        self.debug = config.get('debug')

        self.theta_markers = config.get('theta_by_markers')

    def evaluate(self, model, theta_df):
        sim_data = self.simulate(theta_df)
        theta_hats = [model.estimate_theta(sample) for sample in sim_data]
        plots.generative_results(theta_df, theta_hats)
        return theta_hats

    def simulate(self, theta_df):
        if self.sim_type == 'latent':
            sim_data = self.latent_variables_simulation(theta_df)
        elif self.sim_type == 'mix_dist':
            sim_data = self.mixed_distribution_simulation(theta_df)
        else:
            raise ValueError("Invalid simulation argument.")
        return [build_profile(profile, self.test.index) for profile in sim_data]

    def latent_variables_simulation(self, theta):
        """
        Simulate plasma cfDNA sample based on the read mixture model generation process, where for every read we
        first draw its generating process, and then sample the read by that process.

        Repeat for num_simulations:
            1. Decide randomly the depths in each marker, by given depth distribution (config).
            2. Sample latent variables (which tissue to generate reads from) by Categorical(theta).
            3. Generate U/M/X reads from every tissue by latent variables selection, by the atlas supplied.
            4. aggregate and return UMX profile.

        :param theta: theta vector df, indexed by tissue names, with distribution as values.
        :return: np.array shaped [num_sim, N_markers, MULTINOM_DIM]
        """
        num_markers = self.test.dim[0]
        markers = self.test.index
        P = {tissue: self.test.get_tissue(tissue, 'dist') for tissue in self.test.tissues}

        samples = []
        for _ in np.arange(self.num_sim):
            depths = sample_windows_depths(num_markers, *self.depth_args, self.rng)
            Z = self.sample_latent_variables(depths, theta)
            sample = np.zeros((num_markers, MULTINOM_DIM))
            for idx, i in enumerate(markers):
                for tissue in self.test.tissues:
                    sample[idx] += self.rng.multinomial(n=Z.loc[i, tissue], pvals=P[tissue].loc[i])
            samples.append(sample)
        return np.array(samples)

    def sample_latent_variables(self, depths, theta):
        """
        Generates Matrix of latent variables, by sampling method given (config).
        if random: choose randomly by theta for each window (Theta as probability).
        if deterministic: given depth per marker, select tissues by proportion in theta (as frequency table).
        :param depths: vector of N_marker contains the number of reads required per window.
        :param theta: df containing tissues as index and simplex vector as value.
        :return: Z if self.latent_selection != skip, else df.
        """
        if self.theta_markers == 'estimate':
            Z = self.sample_by_depth_dist(depths, theta)
        elif self.theta_markers == 'uniform':
            Z = self.sample_uniform_markers(depths, theta)
        else:
            raise ValueError("Invalid argument")
        return Z

    def sample_by_depth_dist(self, depths, theta):
        # normalizes theta over depth distribution (if tissue-marker depth distribution is not uniform).
        theta_markers = theta.values[:, 0] * self.test.depth_distribution()
        num_markers = theta_markers.shape[0]
        Z = []
        if self.sample_latent == 'random':
            for i in np.arange(num_markers):
                Z.append(self.rng.multinomial(n=depths[i], pvals=theta_markers.iloc[i]))
        elif self.sample_latent == 'deterministic':
            Z = np.round(theta_markers * depths)
        else:
            raise ValueError("Invalid argument for latent variable selection")
        return pd.DataFrame(Z, index=self.test.index, columns=theta.index.values)

    def sample_uniform_markers(self, depths, theta):
        if self.sample_latent == 'random':
            Z = self.rng.multinomial(n=depths, pvals=theta.values[:, 0])
        elif self.sample_latent == 'deterministic':
            Z = np.round(depths * theta.values).T
        else:
            raise ValueError("Invalid argument for latent variable selection")
        Z = pd.DataFrame(Z, index=self.test.index, columns=theta.index.values)
        # Plot analysis of whether simulation coincide with theta (noised by depths)
        plots.theta_simulation_consistency(Z, depths, theta) if self.debug else None
        return Z

    def mixed_distribution_simulation(self, theta):
        """
        By the initial generative model, considering data point as a complete methylation profile.
        This approach uses direct formulation of mixture model without latent variables. Namely:

            1. Given theta and K distributions (from atlas), the probability of a data point is:
                    f(x) = sum{ theta_k * G(x | p_k) }
                Namely, weighted average of probabilities.
            2. we use this distribution to sample multinomialy per window, given sampled depths.

        Get parameters of generative model (multinomial U,M,X with N_reads trials), to generate nu,nm,nx per window.
        @return methylation profile of dim (n_markers x 3).
        """
        num_markers = self.test.dim[0]
        mixed_distribution = self.test.marker_multinom_params(theta)
        depths = sample_windows_depths(num_markers, *self.depth_args, self.rng)
        result = []
        for i in np.arange(num_markers):
            result.append(self.rng.multinomial(depths[i], mixed_distribution[i], self.num_sim))
        return np.swapaxes(np.array(result), 0, 1)


def sample_windows_depths(num_markers, mean, std, rng):
    """
    dept is defined as number of reads per window. Namely, in a real blood and plasma sample,
    sum_of_reads = depth = U+M+X.
    :return: vector of size num_windows,
    """
    return np.round(rng.normal(mean, std, size=num_markers)).astype(int)
