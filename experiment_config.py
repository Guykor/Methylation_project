import numpy as np
import logging


class Config:
    """
    Designed as a class to allow on-demand changes to the object's value (support notebook enviroment).
    """

    def __init__(self):
        self.marker_file = r"../data/m.bed"
        self.train_path = r"../data/tables/train/"
        self.test_path = r"../data/tables/test/"
        self.blood_path = r"../data/tables/blood_bg/"
        self.theta_path = r"../data/prior_cfDNA.x25.1297.326Kb.l3.csv"

        self.pathologies_paths = {
            'healthy': "..\data\pathologies\healthy_cfdna\\",
            'covid_19': "..\data\pathologies\covid\\",
            'liver_cancer': "..\data\pathologies\liver\\",
        }

        self.WBC_NAME = 'White_Blood_Cells'

        self.ATLAS_SUFFIX = "\\.[\d\.]*"
        self.MARKER_INDEX = 'startCpG'

        self.P_PSEUDO_CNTS = 1
        self.THETA_PSEUDO_CNTS = 0.01

        # Admixture Simulation params
        self.admixture_params = {
            'wbc_name': self.WBC_NAME,

            # 'alphas': list(np.arange(0, 0.11, 0.01)),
            # 'labels': ["0%", '', '', '', '', '5%', '', '', '', '', "10%"],
            # 'subset_problem': ['subset_markers', 'all_markers', 'full_problem'][1],
            # 'num_blood_profiles': 23,
            # 'num_profile_rep': 5,
            # 'required_coverage': 30.,
            # 'sampling_method': ['uniform', 'multinom'][0],
            # 'debug': False,
            #
            # DEBUG MODE
            'alphas': list(np.arange(0, 0.11, 0.05)),
            'labels': ["0%", '5%', "10%"],
            'subset_problem': ['subset_markers', 'all_markers', 'full_problem'][1],
            'num_blood_profiles': 1,
            'num_profile_rep': 2,
            'required_coverage': 40.,
            'sampling_method': ['uniform', 'multinom'][0],
            'debug': False,
        }

        # Generative simulation params
        self.gen_sim_params = {
            'wbc_name': self.WBC_NAME,
            'num_simulations': 100,
            'depth_mean': 30,
            'depth_std': 2,
            'simulation': ['latent', 'mix_dist'][1],
            'sample_latent': ['random', 'deterministic'][0],
            'theta_by_markers': ['uniform', 'estimate'][0],
            'debug': False,
        }


config = Config()
