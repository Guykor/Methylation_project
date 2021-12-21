from scipy.stats import ttest_ind
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class RealDataEvaluator:
    def __init__(self, loader, config):
        self.healthy = loader.load_mixed_samples(config.pathologies_paths['healthy'])
        self.covid19 = loader.load_mixed_samples(config.pathologies_paths['covid_19'])
        self.liver_cancer = loader.load_mixed_samples(config.pathologies_paths['liver_cancer'])

    def evaluate(self, model):
        theta_healthy = estimate_samples(model, self.healthy)
        theta_healthy['sample'] = 'healthy'

        theta_covid = estimate_samples(model, self.covid19)
        theta_covid['sample'] = 'covid_19'

        theta_cancer = estimate_samples(model, self.liver_cancer)
        theta_cancer['sample'] = 'liver_cancer'

        all_estimations = pd.concat([theta_healthy, theta_covid, theta_cancer], axis=0)
        plot_pathologies_estimation(all_estimations)
        all_samples = self.healthy + self.covid19 + self.liver_cancer

        print("*\nModel significance test:")
        print(model_permutation_test(model, all_samples, all_estimations, permute_level='label', num_permutations=5))
        print("*\n Identify disease signal in tissue test")
        lungs_tissues = ['Lung-Ep-Alveo', 'Lung-Ep-Bron']
        liver_tissues = ['Liver-Hep']
        print("Covid19")
        for tissue in lungs_tissues:
            t_pval, perm_pval = identify_pathology_tissues(theta_healthy[tissue], theta_covid[tissue], tissue)
            print(f"T test: {t_pval} < 0.05: {t_pval < 0.05};  Permutation test: {perm_pval} < 0.05: {perm_pval < 0.05}")
        print("*\nLiver Cancer")
        for tissue in liver_tissues:
            t_pval, perm_pval = identify_pathology_tissues(theta_healthy[tissue], theta_cancer[tissue],  tissue)
            print(f"T test: {t_pval} < 0.05: {t_pval < 0.05};  Permutation test: {perm_pval} < 0.05: {perm_pval < 0.05}")
        return all_estimations

def estimate_samples(model, samples):
    results = pd.DataFrame(index=model.atlas.tissues)
    for i, profile in enumerate(samples):
        theta_hat = model.estimate_theta(profile)
        results[i] = theta_hat.values
    return results.T





def identify_pathology_tissues(tissue_healthy_est, tissue_disease_est, tissue_name):
    t_p_val = t_test(tissue_healthy_est, tissue_disease_est)
    df = pd.concat([tissue_healthy_est, tissue_disease_est], axis=1)
    df.columns = ['healthy', tissue_name]
    perm_p_val = permutation_test(df, num_permutation=3000)
    return t_p_val, perm_p_val


def t_test(tissue_healthy_est, tissue_disease_est):
    """
    Student's-T test: compare means of a disease related tissue. Null: theta_k_healthy >= theta_k_disease.
    """
    stat, p_val = ttest_ind(tissue_healthy_est, tissue_disease_est, alternative='less')
    return p_val


def permutation_test(df, num_permutation):
    """
    Permutation test: healthy\disease is var X, tissue-k estimate are Var Y; Null: P(XY) = P(X)P(Y)
    """
    # compare for single tissue: columns ['healthy', 'pathology name']

    df = df.melt()
    rng = np.random.default_rng()
    actual_diff = np.diff(df.groupby('variable').mean(), axis=0).item()
    avg_diffs = []
    for i in np.arange(num_permutation):
        df['variable'] = rng.permutation(df['variable'])
        diff = np.diff(df.groupby('variable').mean(), axis=0).item()
        avg_diffs.append(diff)

    null_dist = np.array(avg_diffs)
    p_val = (null_dist > actual_diff).mean()
    return p_val


def permute_samples_data(samples, kind, num_permutations):
    rng = np.random.default_rng()
    res = []
    axis = 1 if kind == 'labels' else 0  # permute markers per label
    for sample in samples:
        sample = sample.drop(columns='N')
        for _ in np.arange(num_permutations):
            perm = pd.DataFrame(rng.permutation(sample, axis=axis), index=sample.index, columns=sample.columns)
            res.append(perm)
    return res


def model_permutation_test(model, samples, samples_estimates, permute_level, num_permutations):
    """
    theta_hat | healthy = F
    theta_hat | random = G
    Null: F = G
    Ks test as well
    :param model:
    :param samples:
    :param samples_estimates:
    :param permute_level:
    :param num_permutations:
    :return:
    """
    random = permute_samples_data(samples, kind=permute_level, num_permutations=num_permutations)
    rand_est = estimate_samples(model, random)
    result = {}
    for tissue in rand_est.columns:
        stat, p_value = ks_2samp(samples_estimates[tissue], rand_est[tissue], alternative='two-sided')
        result[tissue] = [p_value, p_value < 0.05]

    result = pd.DataFrame(result).T
    result.columns = ['p_value', '< 0.05']
    return result
