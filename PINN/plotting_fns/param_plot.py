# File:         PINN/plotting_fns/param_plot.py
# Usage:        PINN.pl.<fn_name>
# Description:  Visaulizing the dynamics behavior params for high dimensional modeling.
#               Some of the functions are shared between Trajectory-Dependent and Trajectory-Independent Modeling


import os
import numpy as np
import pandas as pd
import torch 
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import seaborn as sns
from .density_plot import umap_by_time
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# predict

def format_ay(array):
    formated  = [np.format_float_scientific(u, precision=2) for u in array]
    return formated

def params_in_umap(adata, prediction, timepoints=None, param='u', copy=True, cell_of_t=True, log=False, clipping=None, subplot_kws=None, umap_kws=None):    
    r"""
    Visaulize the fitted behavior params in umap and by time

    Arguments
    ----------
    adata : AnnData
    prediction : [tensor, ndarray] , the prediction of shape (n_timepoints, n_cells)
    timepoints : list of real-time
    param : str, the param to visaulize, must be one of ['u', 'g', 'v', 'D']
    copy : bool, default to True, will save the params to adata.obs if copy is set to False
    cell_of_t : bool, default to True, only visualize cells of each timepoints. 
                If set to False, all cells will be shown in each panels.

    Example
    ----------
    >>> param = 'g'
    >>> u_pred = Model.predict_param(DataSet=train_DS_t5, param=param)
    >>> adata = train_DS_t5.adata
    >>> params_in_umap(adata, u_pred, param=param)
    """

    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    print("prediction of shape", prediction.shape)
    u_min_ls = format_ay(prediction.min(axis=1))
    u_max_ls = format_ay(prediction.max(axis=1))

    if copy:
        adata = adata.copy()

    if timepoints is None:
        timepoints = adata.uns['pop']['t'][:prediction.shape[0]]

    if clipping is not None:
        assert len(clipping) == 2, "the format of clipping threshold should be"
    
    for i, t in enumerate(timepoints):
        adata.obs[f'Day{t}_{param}'] = prediction[i]

    fig, axs = umap_by_time(lambda x: f'Day{x}_{param}', adata, timepoints, cell_of_t=cell_of_t, subplot_kws=subplot_kws, umap_kws=umap_kws)
    
    if len(timepoints) == 1:
        axs = [axs]

    for i, ax in enumerate(axs):
        title = ax.get_title()
        new_title = title + "\nmin:%s"%u_min_ls[i] + "\nmax:%s"%u_max_ls[i]
        ax.set_title(new_title)
    return fig, axs

def contour_animation(s, continous_u , save_path, fill=False, fps=5):
    r"""
    animation of density contour change by time
    s: nparray, (ngrid**2, 2) , s from train_DS
    continous_u : nparray, (n_timepoints, ngrid**2)
    save_path : str
    """
    fig , ax = plt.subplots(1,1, dpi=300)
    ax.set_xlim(0,1)         
    ax.set_ylim(0,1)

    n_grid = 50

    XX = s[:,0].reshape(n_grid,n_grid)    # in DS, meshgrid is flatten
    YY = s[:,1].reshape(n_grid,n_grid)

    def init():
        Z_ub = continous_u[0].reshape(n_grid, n_grid)
        if fill:
            ax.contourf(XX,YY, Z_ub, cmap='Blues')
        else:
            ax.contour(XX,YY, Z_ub, cmap='Blues')
        ax.set_title("Day 0")

    def run(data):
        if data>0:
            ax.clear()   
            t = np.arange(0, continous_u.shape[0])[data]
            Z_ub = continous_u[data].reshape(n_grid, n_grid)
            if fill:
                ax.contourf(XX,YY, Z_ub, cmap='Blues')
            else:
                ax.contour(XX,YY, Z_ub, cmap='Blues')
            ax.set_title("Day %d"%t)
        else:
            pass

    ani = animation.FuncAnimation(fig, run, frames=continous_u.shape[0], interval=10, init_func=init)  # 製作動畫
    ani.save(save_path, fps=fps, writer='pillow') 


def truncated_clustermap(matrix, num_clusters, truncate_mode="level", p=3, method='ward', cmap='viridis', context_kws={}, show_log=False, cluster_colors=None):
    """
    Create a truncated clustermap with colored dendrogram and return cluster assignments and reordered indices.

    Parameters:
    - matrix: numpy array, the input matrix where rows are to be clustered.
    - num_clusters: int, the number of clusters to form.
    - truncate_mode: str, the truncation mode for the dendrogram (default is "level").
    - p: int, the truncation parameter (e.g., number of levels for "level" mode).
    - method: str, the linkage method to use (default is 'ward').
    - cmap: str, the colormap for the heatmap (default is 'viridis').

    Returns:
    - clusters: numpy array, cluster assignments for each row.
    - reordered_indices: numpy array, reordered row indices based on the dendrogram.
    """
    # Compute pairwise distances and linkage matrix
    row_distances = pdist(matrix, metric='euclidean')
    row_linkage = linkage(row_distances, method=method)
    
    # Form clusters
    clusters = fcluster(row_linkage, num_clusters, criterion='maxclust')
    
    # Create a truncated dendrogram to get reordered indices
    dnd = dendrogram(row_linkage, truncate_mode=truncate_mode, p=p, no_plot=True)
    reordered_indices = dnd['leaves']
    
    # Map cluster labels to colors
    if cluster_colors is None:
        cluster_colors = sns.color_palette("husl", num_clusters)  # Use a color palette
    row_colors = [cluster_colors[label - 1] for label in clusters]  # Map labels to colors
    
    # Create a clustermap with the reordered indices and row colors
    
    show_matrix = np.log(matrix+1e-30) if show_log else matrix


    with plt.rc_context(context_kws):
        g = sns.clustermap(
            show_matrix,
            row_linkage=row_linkage,
            col_linkage=None,  # Only cluster rows
            col_cluster=False,
            cmap=cmap,
            row_colors=row_colors,  # Color rows by cluster
            dendrogram_ratio=(0.2, 0),  # Adjust dendrogram size
            figsize=(8, 8)
        )
    # g.ax_heatmap.set_xticks(range(0,100,10))
    # g.ax_heatmap.set_xticklabels(range(0,100,10))

    # Add a legend for the row colors
    legend_patches = [
        Patch(color=cluster_colors[i], label=f"Cluster {i + 1}")
        for i in range(num_clusters)
    ]
    plt.legend(
        handles=legend_patches,
        title=False,
        bbox_to_anchor=(-0.5, -4),
        loc='lower left',
        borderaxespad=0.
    )

    plt.show()
    
    return clusters, reordered_indices, g