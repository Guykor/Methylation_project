import numpy as np
from scipy import optimize
import pandas as pd

U_IDX = 0
M_IDX = 1
X_IDX = 2


class ProfileMixtureModel:
    def __init__(self, atlas, theta_obj):
        self.theta_obj = theta_obj
        self.atlas = atlas

    def estimate_theta(self, data, label):
        """
        data is a methylation profile (counts) of a blood sample df (n_markers, 3).
        the optimizer tries to minimize the log likelihood using its gradient.
        """
        dim_theta = self.theta_obj.values.shape[0]
        cons_sum = optimize.LinearConstraint(np.ones(dim_theta), 1, 1, True)
        cons_prob = optimize.LinearConstraint(np.eye(dim_theta), np.zeros(dim_theta), np.ones(dim_theta), True)
        result = optimize.minimize(self.log_likelihood,
                                   self.theta_obj.values,
                                   args=(data, self.atlas),
                                   jac=self.log_likelihood_grad,
                                   constraints=[cons_sum, cons_prob]).x

        return self.theta_obj.format(result[:, None])

    def log_likelihood(self, theta, data, atlas):
        """
        Likelihood of the mixture model.
        data is the blood sample matrix of (n_markers, 3), with U/M/X counts per marker.
        """
        multinom_probs = atlas.marker_multinom_params(theta)
        u_theta, m_theta, x_theta = multinom_probs.T
        log_u_theta, log_m_theta, log_x_theta = np.log(u_theta), np.log(m_theta), np.log(x_theta)
        NU, NM, NX = data.U, data.M, data.X
        return -np.sum([np.dot(log_u_theta, NU), np.dot(log_m_theta, NM), np.dot(log_x_theta, NX)], axis=0)

    def log_likelihood_grad(self, theta, data, atlas):
        """
        data is the blood sample matrix of (n_markers, 3), with U/M/X counts per marker.
        without reg term Z (falls in derivation)
        @returns: vector of size num_tissues.
        """
        multinom_probs = atlas.marker_multinom_params(theta)
        u_theta, m_theta, x_theta = multinom_probs.T
        norm_win = lambda tbl, factor: (tbl.values / factor[:, None])
        normed_U, normed_M, normed_X = norm_win(atlas.Pu, u_theta), \
                                       norm_win(atlas.Pm, m_theta), \
                                       norm_win(atlas.Px, x_theta)
        NU, NM, NX = data.U, data.M, data.X
        result = -np.sum([np.dot(normed_U.T, NU), np.dot(normed_M.T, NM), np.dot(normed_X.T, NX)], axis=0)
        return result
