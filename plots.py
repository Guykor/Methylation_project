import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from experiment_config import config
from functions import transform_distribution_df

sns.set_style('whitegrid')


# Descriptive Analysis#


def markers_importance_heatmap(markers, atlas):
    markers['direction'] = 'U'
    markers['direction'][(markers['cell_avg'] - markers['other_avg'] > 0)] = 'M'

    df = markers[[config.MARKER_INDEX, 'direction']].join(markers.significant_cell.explode())
    df = df.sort_values(by=['direction', 'significant_cell']).drop_duplicates(config.MARKER_INDEX)
    sorted_markers = df[config.MARKER_INDEX]
    data = atlas.Pu.loc[sorted_markers].reset_index(drop=True)

    f = plt.figure(figsize=(12, 8))
    ax = sns.heatmap(data, cmap=sns.diverging_palette(220, 20, s=65, l=65, as_cmap=True),
                     yticklabels=300, cbar_kws={'label': "Non-methylated DNA rate"})
    ax.figure.axes[-1].yaxis.label.set_size(13)
    ax.set_xlabel("Cell Types", fontsize=13)
    ax.set_ylabel("Markers", fontsize=13)
    ax.tick_params('both', labelsize=11)
    plt.show()
    return f


def plot_theta(theta_df, kind):
    """
    Plot current values of theta for each tissue.
    """
    sns.set_style('whitegrid')
    if kind == 'pie':
        f, ax = plt.subplots(figsize=(8, 8))
        prop = theta_df.sort_values(by='value')
        prop = prop[prop.value >= 0.01]
        prop.loc['Other'] = 1 - prop.sum()
        prop.plot.pie(y='value', legend=False, autopct='%1.1f%%',
                      colors=sns.color_palette('Reds'), fontsize=14, ax=ax)
        ax.set_ylabel("")
        plt.show()
        return f

    if kind == 'bar':
        f, ax = plt.subplots(figsize=(12, 5))
        sns.set_theme(style="whitegrid")
        sns.barplot(data=theta_df.reset_index(), x='tissue', y='value', ax=ax, color='lightblue')
        ax.set_xlabel("")
        ax.set_ylabel("")
        f.autofmt_xdate()
        f.tight_layout()
        plt.title("Tissue Proportions")
        plt.show()
        return f


def depth_by_tissues(atlas_depth_df):
    tissue_depths = atlas_depth_df.sum()

    f, ax = plt.subplots(figsize=(12, 5))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=tissue_depths.reset_index(), x='name', y=0, ax=ax, color='gray')

    ax.set_xlabel("")
    ax.set_ylabel("Million reads")
    f.autofmt_xdate()
    f.tight_layout()
    plt.title("Train data reads per tissue")
    plt.show()


def depth_by_marker(atlas_depth_df):
    f = plt.figure()
    atlas_depth_df["Sum observations per marker"] = atlas_depth_df.sum(axis=1)
    atlas_depth_df[["Sum observations per marker"]].boxplot()
    plt.title("Observations per marker over tissues")
    f.autofmt_xdate()
    f.tight_layout()
    plt.show()


def depth_exploration(depths, V):
    # Cell type distribution across markers
    sns.catplot(data=depths.melt(), x='name', y='value', kind='box', height=6, aspect=2)
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.xticks(rotation=90, fontsize=14)
    plt.show()

    # Cell type distribution across sig and unsignificant marekers
    scoped = depth_by_scope(depths, V)
    sns.catplot(data=scoped, x='value', y='tissue', hue='scope', kind='violin', split=True, height=6)
    plt.ylabel("")
    plt.xlabel("")
    plt.yticks(fontsize=13)
    plt.show()


def depth_by_scope(depths, V):
    rel_marks = V != 0
    depths_in = (depths * rel_marks).melt(var_name='tissue').query('value != 0')
    depths_in['scope'] = 'relevant_markers'
    depths_out = (depths * (1 - rel_marks)).melt(var_name='tissue').query('value != 0')
    depths_out['scope'] = 'others'
    depths_by_scope = pd.concat([depths_in, depths_out], axis=0)
    return depths_by_scope


def markers_dist(markers):
    """counts significant markers per cell type, by their direction (methyated/unmethylated)"""
    markers['direction'] = 'Unmethylated'
    markers['direction'][(markers['cell_avg'] - markers['other_avg'] > 0)] = 'Methylated'

    markers_dist = markers[['startCpG', 'direction']].join(markers.significant_cell.explode())
    markers_dist = markers_dist.groupby(['significant_cell', 'direction']).nunique()
    markers_dist.columns = ['num_markers']
    df = markers_dist.reset_index().pivot(index='significant_cell', columns='direction', values='num_markers').fillna(0)
    df.plot.bar(stacked=True, figsize=(8, 5))
    plt.ylabel("Num Markers")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()


def plot_distribution_by_markers(tissue_dist_df):
    if tissue_dist_df.shape[0] > 200:
        return
    tissue_dist_df['U'] = tissue_dist_df['U'] / tissue_dist_df['N']

    sns.set_style('whitegrid')
    f, ax = plt.subplots()
    tissue_dist_df['U'].reset_index(drop=True).plot.bar(ax=ax)
    plt.tight_layout()
    plt.show()


# Simulation Plots #

# Mix
def admixture_cell_results(cell_df, cell_type, labels):
    alphas = cell_df.columns
    ax = cell_df.boxplot(figsize=(4, 4))
    ax.grid(False)
    locs = ax.get_xticks()
    ax.plot(locs, alphas)
    ax.set_title(cell_type)

    ax.set_xticklabels(labels)
    ax.set_yticks(alphas)
    ax.set_yticklabels(labels)
    plt.show()
    return ax

def admixture_results(results, label, alphas, alphas_labels):
    num_subplots = len(results)
    subplots_per_row = 3 if num_subplots > 3 else num_subplots
    rows = num_subplots // subplots_per_row
    rows += num_subplots % subplots_per_row
    position = range(1, num_subplots + 1)

    f = plt.figure(1)
    for i, (cell_type, result) in enumerate(results.items()):
        ax = f.add_subplot(rows, subplots_per_row, position[i])
        results[cell_type].boxplot(ax=ax)
        locs = ax.get_xticks()
        ax.plot(locs, alphas)
        ax.set_title(cell_type)
        ax.set_xticklabels(alphas_labels)
    #     ax.set_ylim(-0.01, 0.11)
    #     ax.set_yticklabels(xlabels)
    f.suptitle(label)
    plt.tight_layout()
    plt.show()
    return f


def mix_profile_check(profiles, V, cell_type, alpha):
    cell_type_marks = V[cell_type] != 0
    for profile in profiles:
        sim = transform_distribution_df(profile).drop(columns='N')
        p = sim.melt(ignore_index=False, var_name='type')
        sim['cell_markers'] = 0
        sim['cell_markers'][cell_type_marks] = 1
        df = pd.merge(sim['cell_markers'], p, left_index=True, right_index=True)

        sns.catplot(data=df, x='type', y='value', hue='cell_markers', kind='box')
        plt.title(f'Mix_sim, {cell_type}, alpha={alpha}')
        plt.ylabel("")
        plt.xlabel("")
        plt.show()


def mix_theta_check(alpha, cell_type, cell_reads, blood_reads):
    df = pd.concat([cell_reads, blood_reads], axis=1)
    df.columns = [cell_type, 'whole_blood']

    mean_over_marks = np.mean(df.T / df.sum(axis=1), axis=1)
    mean_over_marks = mean_over_marks.rename('marker_mean')

    ratio_all_reads = df.sum(axis=0)
    ratio_all_reads /= ratio_all_reads.sum()
    ratio_all_reads = ratio_all_reads.rename('total_freq')

    theta = pd.DataFrame([alpha, 1 - alpha], index=[cell_type, 'whole_blood'], columns=['theta'])

    theta_comp = pd.merge(mean_over_marks, ratio_all_reads, left_index=True, right_index=True)
    theta_comp = pd.merge(theta_comp, theta, left_index=True, right_index=True)
    theta_comp.index.name = 'tissue'
    theta_comp = theta_comp.reset_index().melt(id_vars='tissue')

    sns.barplot(data=theta_comp.reset_index(), x='tissue', y='value', hue='variable')
    plt.title('Simulated profile freq from sum read vs theta')
    plt.show()

    plt.show()


# Gen
def L2_error(theta_df, predictions):
    L2 = np.array([np.power(theta_df - predictions[i], 2) for i in np.arange(len(predictions))])[:, :, 0]
    df = pd.DataFrame(L2, columns=theta_df.index)
    return df.melt()


def weighted_error(theta_df, predictions):
    """measure mistakes for each coordinate"""
    weighted = np.array([(theta_df - predictions[i]) / theta_df for i in np.arange(len(predictions))])[:, :, 0]
    df = pd.DataFrame(weighted, columns=theta_df.index)
    return df.melt()


def plot_error_results(error_df, name, num_sim):
    sns.catplot(x='tissue', y='value', kind='box', data=error_df)
    # error_df.mean().plot(linestyle='', marker=".", c='r', yerr=error_df.std(),
    plt.title(f'{name} Error over {num_sim} simulations')
    plt.xlabel("")
    plt.tight_layout()
    plt.show()


def generative_results(theta_df, predictions):
    # plot_error_results(L2_error(theta_df, predictions), 'L2', len(predictions[0]))
    # plot_error_results(weighted_error(theta_df, predictions), 'Weighted', len(predictions[0]))

    sns.set_style(style="dark")
    f, ax = plt.subplots(figsize=(12, 5))
    df = pd.DataFrame(np.array(predictions)[:, :, 0], columns=theta_df.index)
    sns.boxplot(data=df, ax=ax)
    plt.setp(ax.artists, edgecolor='k', facecolor='w')
    plt.setp(ax.lines, color='k')

    sns.barplot(data=theta_df.reset_index(), x='tissue', y='value', ax=ax, color='lightblue', label='True')
    ax.set_ylabel("")

    f.autofmt_xdate()
    f.tight_layout()
    plt.show()


def theta_simulation_consistency(Z, depths, theta_df):
    """
    Plot analysis of whether simulation coincide with tissues relative contribution.
    :param Z: df of size N_markers, tissues, s.t cell i,j is the number of reads out of depth[i] that came from
    tissue j in the simulation.
    :param depths: vector of depths per window (number of reads, or cfDNA)
    :param theta_df: df theta by which the simulation is being made.
    :return: plots histogram of frequencies for every tissue over all markers, and prints overall frequency of
    entire read from each tissue.
    """
    total = round(Z.sum() / Z.sum().sum(), 3)
    plt.bar(total.index, total.values)
    plt.plot(theta_df.index, theta_df.values[:, 0], c='red', marker='D', linestyle="")
    plt.title('tissue relative part, all reads, vs theta')
    plt.show()

    f, ax = plt.subplots(figsize=(8, 5))
    axes = (Z.T / depths).T.hist(ax=ax).flatten()
    for i, tissue in enumerate(Z.columns):
        axes[i].axvline(x=theta_df.loc[tissue].value, c='red')
    f.suptitle("frequency of fraction for every tissue")
    plt.show()
