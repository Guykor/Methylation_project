import os
import pandas as pd
import numpy as np

from experiment_config import config
import warnings
from project_logger import get_logger
import json
import pickle
from datetime import datetime
import os
from shutil import copyfile

warnings.simplefilter("ignore")

MULTINOM_DIM = 3
PROFILE_HEADER = ["chromosome", "start", "end", "startCpG", "endCpG", "U", "X", "M"]
MARKER_HEADER = ["chromosome", "start", "end", "startCpG", "endCpG", "significant_cell", 'num_sites', 'num_bps',
                 'cell_avg', 'other_avg']

logger = get_logger(__name__)


def get_tissues_files(path, suffix_remove=None):
    paths = {}
    for file in os.listdir(path):
        if file.endswith(".gz"):
            tissue_name = file.split(".uxm")[0]
            paths[tissue_name] = path + file
    paths = pd.DataFrame({'name': paths.keys(), 'path': paths.values()})
    if suffix_remove:
        paths.name = paths.name.str.replace(suffix_remove, '')
    return paths


def select_by_tissue(paths, required_tissues):
    """
    Assuming that the required_tissues, supplied by the init_theta, coincide with the profiles file_names.
    :param paths:
    :param required_tissues:
    :return:
    """
    matches = paths.name.isin(required_tissues)
    filtered_out = paths.name[~matches]
    file_not_found = required_tissues[~np.isin(required_tissues, paths.name)]

    logger.info(f"{len(filtered_out)} filtered out.")
    if len(file_not_found):
        logger.warning(f"{len(file_not_found)} files are not in folder: {file_not_found}")
    return paths[matches]


def read_profiles(paths, pseudo_counts, index):
    """
    read profiles, add pseudo counts and return dict of filename-> counts df.
    :param index_align:
    :param pseudo_counts: additive pseudo counts to add for all categories(window/label).
    :param paths: pd.DataFrame(name, paths)
    """
    profiles = {}
    for name, path in dict(paths.values).items():
        df = read_profile(name, path, index)
        df = add_pseudo_counts(df, pseudo_counts)
        profiles[name] = df
    return profiles


def read_profile(name, path, global_index):
    """
    reads a single methylation profile, while validating global indexer by markers_df.
    :param name: profile name (tissue)
    :param path: path to file.
    :param global_index: index values and order for all data
    :return: df of size (N_subset_markers x 3)
    """
    profile = pd.read_csv(path, sep='\t', names=PROFILE_HEADER).set_index(config.MARKER_INDEX)
    try:
        return profile.loc[global_index, ['U', 'M', 'X']]
    except KeyError:
        logger.error(f"{name} index mismatch")
        raise IndexError


def add_pseudo_counts(df, pseudo_cnts):
    df.U = (df.U + pseudo_cnts)
    df.M = (df.M + pseudo_cnts)
    df.X = (df.X + pseudo_cnts)
    df['N'] = df.U + df.M + df.X
    return df


def build_profile(np_profile, index):
    df = pd.DataFrame(np_profile, index=index, columns=['U', 'M', 'X'])
    df['N'] = df.sum(axis=1)
    return df


def transform_distribution(tissues_data):
    return {name: transform_distribution_df(df.copy()) for name, df in tissues_data.items()}


def transform_distribution_df(df):
    """
    transforms every tissue df to distribution by marker.
    for marker i and tissue j we compute P(Uij) = U_ij + delta / N
    where U_ij is the U marked reads for tissue j in marker i, and N = U_ij + M_ij + X_ij + (3 * delta)
    """
    df.U /= df.N
    df.M /= df.N
    df.X /= df.N
    return df


def transform_umx_tables(tissues_dist: dict, tissue_ordering):
    """
    Transforms K dfs of dim {N x 3} to 3 matrices of dim {N x K}, s.t the columns of every matrix is ordered by
    tissue_ordering.
    **Note: To be used after Marker index and tissues presence validation amongst all dfs.**
    :param tissues_dist: dictionary: {tissue name -> UMX distribution per marker df}.
    :param tissue_ordering: Series or np array that contains tissue names by some order.
    :return: U/M/X df where rows are markers and columns are the tissues
            (just transformation of format, no change to data).
    """
    res = []
    for type_ in ["U", "M", "X"]:
        tbl = pd.concat([df[[type_]] for df in tissues_dist.values()], axis=1)
        tbl.columns = list(tissues_dist.keys())
        tbl = tbl.reindex(columns=tissue_ordering)
        if np.any(tbl.isna()):
            print(f"Warning: {type_} table contains zero probability.")
        res.append(tbl)
    return res


#### Marker file Handling ####
def load_markers(path, tissues):
    """
    Load markers data frame, filter only relevant markers for given tissues.
    :param path:
    :param tissues:
    :return:
    """
    # at most 25 markers were picked by feature selection per cell type
    markers = pd.read_csv(path, sep='\t', header=None).drop(columns=[6, 11, 12, 13])
    markers.columns = MARKER_HEADER
    markers.significant_cell = markers.significant_cell.str.split(":")

    all_cell_types = markers.set_index(config.MARKER_INDEX).significant_cell.explode()
    relevant_markers = all_cell_types[all_cell_types.isin(tissues)].reset_index()[config.MARKER_INDEX].drop_duplicates()

    if relevant_markers.shape[0] != markers.shape[0]:
        logger.info(f"Markers file had {markers.shape[0]}, "
                    f"after cell-type selection, remaining: {relevant_markers.shape[0]}")

    return markers[markers[config.MARKER_INDEX].isin(relevant_markers)]


def _map_missing_tissues(missing_tissues, marker_tissues):
    """
    maps missing tissues in the significant cell column in marker df, to partial matches in the column with the
    missing tissues and output a df that summarize partial matches.
    if there are still missing tissues that have no match or partial match at all, the function will raise
    ValueError.
    """
    rows = []
    for marker_tissue in marker_tissues.unique():
        for theta_tissue in missing_tissues:
            if marker_tissue in theta_tissue:
                row = [marker_tissue, theta_tissue]
                rows.append(row)
    df = pd.DataFrame(rows)
    df.columns = ['partial_match_markers', 'missing_tissue_name']

    logger.info(f"{len(missing_tissues)} tissues missing, {df.shape[0]} had partial match will be replaced.")
    return df


def align_marker_tissue_names(tissues, marker_tissues):
    """
    Fixes problems with missing tissues in the significant-cell column in the markers file the following manner:
    1. if missing tissues have partial matches with names in the column, we replace the partial string.
    2. if there are duplicate tissues (different types of heart-cell in single category), we join the significant
    markers to a single vector and duplicate it to match the name of each of the subcategories.
    :return: (fixed markers table, dict of remaining issues, missing_tissue -> name in markers for duplicates).
    """
    missing_tissues = tissues[~np.isin(tissues, marker_tissues)]
    if len(missing_tissues) != 0:
        mapping = _map_missing_tissues(tissues, marker_tissues)
        is_dup = mapping.duplicated('partial_match_markers', False)
        dups = mapping.loc[is_dup]

        # Duplicates are treated as a single category, if there are subcategories missing under that level,
        # we will just duplicate the general category significant markers to all subcategories.
        dups['patch'] = dups.missing_tissue_name.str.replace("-.*", '')
        mapping.loc[is_dup] = dups[['partial_match_markers', 'patch']]

        replace_dict = dict(mapping.values)
        to_replace = marker_tissues.isin(replace_dict.keys())
        add_cols = dict(dups[['missing_tissue_name', 'patch']].values)
        marker_tissues[to_replace] = marker_tissues[to_replace].map(replace_dict)
        return marker_tissues, add_cols
    logger.info("V table: No missing tissues")
    return marker_tissues, {}


def create_marker_importance(tissues, markers):
    """"
    Creates V table: binary table of shape [N_markers, K] that maps every tissue to its relevant markers.
    Assumed the marker.significant_cell column has already been fixed with partial matches,
    and add_cols is used to handle multiple partial matches.

    if there are more tissues missing, we log the problem and defines a vector of ones in all windows.
    """
    significant_markers = markers.set_index(config.MARKER_INDEX).significant_cell.explode()
    significant_markers, add_cols = align_marker_tissue_names(tissues, significant_markers)

    significant_markers = significant_markers.reset_index() \
        .groupby(config.MARKER_INDEX).significant_cell.apply(lambda x: ':'.join(x))
    # create V table
    V = significant_markers.str.get_dummies(sep=':')

    for new_col, old_col in add_cols.items():
        V[new_col] = V[old_col]
    if add_cols:
        V = V.drop(columns=add_cols.values())
        logger.info(f"V table: {list(add_cols.keys())} had the same marker mapping, problem fixed.")

    more_missing = tissues[~np.isin(tissues, V.columns)]
    for tissue in more_missing:
        V[tissue] = 1

    V = V.reindex(columns=tissues)  # order column by tissues.
    # validate
    remaining_tissues = tissues[~np.isin(tissues, V.columns)]
    if ~np.all(V.columns == tissues) or len(remaining_tissues) != 0:
        raise ValueError("V table tissue names problem.")
    return V


def validate_theta_dist(theta):
    if np.any(np.isnan(theta)):
        raise ValueError("theta contains Nans.")
    if round(theta.sum().item()) != 1.0:
        raise ValueError("theta does not sum to 1.")
    if np.any(theta) <= 0:
        raise ValueError("theta contains zeros or negative values.")


def validate_atlas_data(Pls, Cls=None):
    mats = Pls + Cls if Cls else Pls
    shape_check(mats)
    nan_check(mats)
    non_zero_check(mats)
    if not np.all(np.round(sum(Pls)) == 1.):
        raise ValueError("Distribution does'nt sums to 1.")


def shape_check(matrices):
    shape = matrices[0].shape
    for mat in matrices:
        if mat.shape != shape:
            raise ValueError("different shapes for Matrices")


def nan_check(matrices):
    for mat in matrices:
        if np.any(np.isnan(mat)):
            raise ValueError("Matrix contains Nans.")


def non_zero_check(matrices):
    for mat in matrices:
        if np.any(mat == 0.):
            raise ValueError("Matrix contains Zeros.")


def save_results(sim_type, results, config, logger_path=None):
    dt_string = datetime.now().strftime(r"%d%m%y_%H%M%S")
    results_folder_name = f'../results/{sim_type}/{dt_string}/'
    os.makedirs(results_folder_name)
    with open(results_folder_name + 'results.pkl', 'wb') as file:
        pickle.dump(results, file)
    if logger_path is not None:
        with open(results_folder_name + 'config.json', 'w') as fp:
            json.dump(config, fp)
        copyfile(logger_path, results_folder_name + logger_path)
