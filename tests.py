from data_api import Loader
from experiment_config import config
from plots import plot_dist_by_markers
from evaluation.admixture import AdmixtureEvaluator


def mix_simulation_distribution_compare_wbc_subcategory():
    loader = Loader(config)
    cell_type = 'Liver-Hep'

    loader.group_wbc(config.WBC_NAME)
    wbc_dist = loader.train_atlas.get_tissue(config.WBC_NAME, 'dist')

    blood_profiles = loader.load_mixed_samples(config.blood_path)
    cell_profile = loader.test_atlas.get_tissue(cell_type, 'cnts')

    admixture_eval = AdmixtureEvaluator(blood_profiles, loader.test_atlas, config.admixture_params, wbc_dist)
    simulations = admixture_eval.simulate_by_coverage(blood_profiles[0], cell_profile, 0.05)
    for sim in simulations:
        plot_dist_by_markers(sim[loader.V[cell_type] != 0])

mix_simulation_distribution_compare_wbc_subcategory()


#
# def test_mix_simulation_dist_compare_wbc():
#     loader = Loader(config)
#     blood_profiles = loader.load_mixed_samples(config.blood_path)
#

#     cell_type = 'Liver-Hep'
#     wbc_type = 'Blood-T'
#     loader.subset_tissues([cell_type, wbc_type], relevant_markers=True)
#
#     admixture_eval = AdmixtureEvaluator(blood_profiles, loader.test_atlas, config.admixture_params)
#     cell_profile = loader.test_atlas.get_tissue(cell_type, kind='cnts')
#     admixture_eval.simulate_by_coverage(0.05, blood_profiles[0], cell_profile)
#
