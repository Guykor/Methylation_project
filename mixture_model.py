import copy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from functions import add_pseudo_counts
from project_logger import get_logger

logger = get_logger(__name__)


class ReadMixture:
    def __init__(self, atlas, init_theta, marker_importance, P_pseudo_cnts, plot_EM):
        self.atlas = atlas
        self.theta_obj = init_theta
        self.V = marker_importance
        self.pseudo_cnts = P_pseudo_cnts

        self.theta_over_time, self.EM_log = None, None
        self.plot_EM = plot_EM

    def subset_data(self, tissues, markers=None):
        V = self.V.loc[markers, tissues] if markers is not None else self.V[tissues]
        atlas = copy.deepcopy(self.atlas).subset(tissues, markers)
        theta = copy.deepcopy(self.theta_obj).subset(tissues)
        return ReadMixture(atlas, theta, V, self.pseudo_cnts, self.plot_EM)

    def estimate_theta(self, data, label=None):
        theta_hat, atlas_P_hat = self._run_EM(data, label)
        return theta_hat

    def _run_EM(self, data, label=None):
        """
        Running EM algorithm for the problem, considering initlal theta and atlas distribution, estimate both by MLE.
        Complete description of the algorithm stages in github.cs.huji.ac.il/guy-korn/Methylation_project.
        :param data: Methilation profile (count data) of dim {N x 3}.

        Required Format:
        1. V: pd.DataFrame shape [N_markers, K].
        2. Nl's and N are np.arrays of shape [N_markers,1].
        3. theta is np.array of shape [K,1]
        4. Pl's are np.arrays of shape [N_markers, K]
        5. Sl's are pd.DataFrames of shape [N_markers, K]
        6. Wl's are np.arrays of shape [N_markers, K]

        :returns updated theta_obj (theta_hat), and atlas (P_hat)
        """
        # Data formatting
        data = add_pseudo_counts(data.copy(deep=True), self.pseudo_cnts)
        N = data.N.values[:, None]
        Nls = data.drop(columns='N').T.values[:, :, None]  # 3 column vectors (N x 1)
        # Initial Params
        theta = self.theta_obj.values
        Pls = self.atlas.Pu.values, self.atlas.Pm.values, self.atlas.Px.values

        # Pre Process Stage #
        # 1. Sl are NxK DataFrames, s.t column k is e.w multiply of Nl * Vk (column vector of V).
        Sls = [Nl * self.V for Nl in Nls]
        # 2. Norm factors per tissue dim K. np.array. equivalent to 1 / sum(Sls).sum().
        d_inv = 1 / (self.V.T @ N).values
        # configure logging
        self._init_EM_logger(Nls, Pls, theta)

        # EM loop
        counter = 1
        while not self.converged(1e-12, 1e-8, counter) and counter <= 40:
            "1. E-step "

            Wls = [responsibilities(Nl, Pl, theta) for Nl, Pl in zip(Nls, Pls)]

            "2. M-step - Theta "

            u_elem, m_elem, x_elem = [theta_nominator_elem(Wl, Sl) for Wl, Sl in zip(Wls, Sls)]
            theta_new = d_inv * (u_elem + m_elem + x_elem)
            # Update theta logs
            self._EM_log_theta(theta, theta_new)
            # update theta
            theta = theta_new

            "3. M-step Pl's "

            Pu_new, Pm_new, Px_new = [Nl * Wl for Nl, Wl in zip(Nls, Wls)]
            # sums element-wise
            normalize = Pu_new + Pm_new + Px_new
            # divide element-wise
            Pls_new = Pu_new / normalize, Pm_new / normalize, Px_new / normalize
            # Update P logs
            self._EM_log_P(Pls, Pls_new)
            # update P
            Pls = Pu, Pm, Px = Pls_new

            "4. Log the log likelihood "

            self._EM_log_like(Nls, Pls, theta)

            "5. Validate Results "

            if np.any(theta <= 0.):
                logger.error("theta contains zero or negative values, have to add pseudo-counts to EM")
            if np.any(normalize == 0.):
                logger.error("divide by zero, have to add pseudo-counts to EM")
            if np.any(Pu == 0.) or np.any(Pm == 0.) or np.any(Px == 0.):
                logger.error("zero counts, have to add pseudo-counts to EM")
            counter += 1

        self._plot_EM_logs(label) if self.plot_EM else ''
        return self.theta_obj.format(theta), self.atlas.format(*Pls)

    def converged(self, epsilon, epsilon_P, iter):
        Theta_converged = self.EM_log['theta_diff'][-1] < epsilon
        P_converged = sum(self.EM_log['Pl_diffs'][-1]) < epsilon_P
        Likelihood_converged = len(self.EM_log['log_likelihood']) > 1 and \
                               np.power(self.EM_log['log_likelihood'][-1] - self.EM_log['log_likelihood'][-2],
                                        2) < epsilon

        if Theta_converged:
            logger.debug(f"Theta_converged at iteration {iter}")
        if P_converged:
            logger.debug(f"P_converged at iteration {iter}")
        if Likelihood_converged:
            logger.debug(f"Likelihood_converged at iteration {iter}")
        return Theta_converged or P_converged

    def _init_EM_logger(self, Nls, Pls, theta):
        self.theta_over_time = [theta]
        self.EM_log = {'theta_diff': [np.nan],
                       'Pl_diffs': [[np.nan, np.nan, np.nan]],
                       'log_likelihood': [log_likelihood(theta, Pls, Nls)]}

    def _EM_log_theta(self, theta, theta_new):
        theta_diff = np.power(theta_new - theta, 2).sum()
        self.EM_log['theta_diff'].append(theta_diff)
        self.theta_over_time.append(theta_new)

    def _EM_log_P(self, Pls, Pls_new):
        Pl_diffs = [np.power(Pl_new - Pl, 2).sum().sum() for Pl_new, Pl in zip(Pls_new, Pls)]
        self.EM_log['Pl_diffs'].append(Pl_diffs)

    def _EM_log_like(self, Nls, Pls, theta):
        self.EM_log['log_likelihood'].append(log_likelihood(theta, Pls, Nls))

    def _plot_EM_logs(self, label=None):
        logs = {}
        logs['Theta_diff'] = self.EM_log['theta_diff']
        logs['Pu_diff'], logs['Pm_diff'], logs['Px_diff'] = np.array(self.EM_log['Pl_diffs']).T
        logs['Log-Likelihood'] = self.EM_log['log_likelihood']
        logs_df = pd.DataFrame(logs)
        P_logs_df = logs_df.iloc[:, 1:-1].reset_index().melt(id_vars='index')
        progress_plot(logs_df, P_logs_df, label)
        theta_ot = pd.DataFrame(np.array(self.theta_over_time)[:, :, 0], columns=self.theta_obj.tissues)
        others = theta_ot.loc[:, theta_ot.std(axis=0) <= 0.001]
        theta_ot['others'] = others.mean(axis=1)
        theta_ot = theta_ot.drop(columns=others.columns)
        theta_ot.plot(title='Theta over time')
        # plt.legend(loc=2, prop={'size': 5})
        plt.show()


def responsibilities(Nl, Pl, theta):
    """
    Nl is np.array column vector shape [N_markers,1]
    Pl is np.array of shape [N_markers, K]
    theta is np.array of shape [K,1]
    :returns Wl, np.array of shape [N_markers, K], with no zero probabilities
    """
    Wl = Nl * Pl * theta.T
    Wl /= Wl.sum(axis=1)[:, None]
    return Wl


def theta_nominator_elem(Wl, Sl):
    """
    Sl is pd.DataFrames of shape [N_markers, K]
    Wl is np.array of shape [N_markers, K]
    """
    return (Wl * Sl).values.sum(axis=0)[:, None]


# likelihood
def like_elem(theta, Pl):
    """
    theta is np.array of shape [K,1]
    Pl is np.array of shape [N_markers, K]
    assuming no sparsity in both arrays.
    returns np.array of shape [N_markers,1]
    """
    return np.log((theta.T * Pl).sum(axis=1))[:, None]


def log_likelihood(theta, Pls, Nls):
    elements = [Nl.T @ like_elem(theta, Pl) for Nl, Pl in zip(Nls, Pls)]
    return sum(elements).item()


def progress_plot(logs_df, P_logs_df, label):
    xs = np.arange(1, logs_df.shape[0]+1)

    sns.set_style("whitegrid")
    f, axs = plt.subplots(1, 3, sharex=True, figsize=(18, 5))

    g1 = sns.lineplot(data=logs_df, x=xs, y='Theta_diff', legend=False, ax=axs[0])
    g1.set_title("Theta Convergence")
    g1.set_ylabel("L2 diff")

    g2 = sns.lineplot(data=P_logs_df, x=P_logs_df['index'].values, y='value', hue="variable", ax=axs[1])
    g2.set_title("P Convergence")
    g2.set_ylabel("L2 diff")

    g3 = sns.lineplot(data=logs_df, x=xs, y='Log-Likelihood', ax=axs[2])
    g3.set_title("Log-Likelihood Convergence")
    [ax.set_xlabel("EM Iteration") for ax in axs]

    f.suptitle(label)
    f.tight_layout()

    # theta_diff_formula = "$\sum_{k=1}^{K}|\\theta^{(t+1)}_k - \\theta^{(t)}_k|$"
    # P_diff_formula = "$\sum_{i}\sum_{k}|[P_{l}^{(t+1)}]_{i,k} - [Pl_{l}^{(t)}]_{i,k}|$"
    return f
