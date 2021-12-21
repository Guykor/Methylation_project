import plots
from evaluation.real_data import RealDataEvaluator
from data_api import Loader
from experiment_config import config

import numpy as np

from functions import transform_distribution, save_results
from mixture_model import ReadMixture
from priors import ThetaGenerator
from evaluation.admixture import AdmixtureEvaluator
from project_logger import setup_logger, FILE_HANDLER

import time

from evaluation.generative import GenerativeEvaluator

logger = setup_logger()

# define loading procedure - enforces same format (markers and tissue) to all data used.
loader = Loader(config)

# By markers alows aggregation that emphasises the data from relevant markers of each cell type
loader.unify_tissues('Pancreas', ['Pancreas-Alpha', 'Pancreas-Beta', 'Pancreas-Delta'], by_markers=True)

# Subset experiment (Select cell types for theta object, train and test atlas and marker relevance table
loader.subset_tissues(['Bladder-Ep', 'Liver-Hep', 'Lung-Ep-Alveo', 'Lung-Ep-Bron', 'Neuron'], relevant_markers=False)

# Subset only markers from data artifacts
liver_rel_markers = (loader.markers_importance['Liver-Hep'] != 0).index.values
loader.subset_markers(liver_rel_markers)

# Specific Grouping method for white blood cells
loader.group_wbc(config.WBC_NAME, by_markers=True)
    

# Define Mixture model #
model = ReadMixture(atlas=loader.train_atlas, init_theta=loader.theta, marker_importance=loader.markers_importance,
                    P_pseudo_cnts=config.P_PSEUDO_CNTS, plot_EM=False)

# Evaluation Procedure #
# Not included in the test. One can decide to estimate wbc dist by whole blood samples.
wbc_dist = loader.train_atlas.get_tissue(config.WBC_NAME, 'dist')
wbc_cnts = loader.train_atlas.get_tissue(config.WBC_NAME, 'cnts')

# Admixture Simulation #
blood_profiles = loader.load_mixed_samples(config.blood_path)
admixture_eval = AdmixtureEvaluator(blood_profiles, loader.test_atlas, config.admixture_params, wbc_dist)


start = time.time()
results = admixture_eval.evaluate(model)
logger.info(f"simulation took  {(time.time() - start) / 60}  minutes.")
save_results('admix', results, config.admixture_params, FILE_HANDLER)

# Generative Simulation #

loader.test_atlas.insert(config.WBC_NAME, wbc_dist.values.T, wbc_cnts.drop(columns='N').values.T)
generative_eval = GenerativeEvaluator(loader.train_atlas, config.gen_sim_params)
print("Real prior")
results = generative_eval.evaluate(model, loader.theta._data)
loader.theta._data.to_csv("baseline_prior.csv")
save_results('generative_real', results, config.gen_sim_params)
print("Uniform Prior")
theta = ThetaGenerator(loader.tissues).uniform()
theta.to_csv("uniform_theta.csv")
results = generative_eval.evaluate(model, theta)
save_results('generative_uniform', results, config.gen_sim_params)

# Hypothesis testing
hypothesis = RealDataEvaluator(loader, config)
estimations = hypothesis.evaluate(model)
save_results('real_estimations', estimations, config)

