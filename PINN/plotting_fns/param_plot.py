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
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.colorbar import ColorbarBase
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

    Return
    ---------
    fig, axs


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

    fig, axs = umap_by_time(lambda x: f'Day{x}_{param}', adata, timepoints, time_mask=cell_of_t, subplot_kws=subplot_kws, umap_kws=umap_kws)
    
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


def truncated_clustermap(matrix, 
                        num_clusters, 
                        original_shape=None, 
                        truncate_mode="level", 
                        p=3, method='ward', 
                        cmap='viridis', 
                        context_kws={}, 
                        cbar_kws={},
                        show_log=False, 
                        cluster_colors=None,
                        col_colorbar_size=(0.02, 0.15), 
                        col_colorbar_location_x = (0.0, 0.3), 
                        col_colorbar_location_y = (0.05, 0.3),  
                        col_colorbar_tick_labels_x=None,
                        col_colorbar_tick_labels_y=None):
    """
    Create a truncated clustermap with colored dendrogram and return cluster assignments and reordered indices.

    Arguments
    ----------
    matrix: numpy array, the input matrix where rows are to be clustered.
    num_clusters: int, the number of clusters to form.
    original_shape: tuple of int, original 2D dimensions (n_rows, n_cols) of the matrix before flattening.
    truncate_mode: str, the truncation mode for the dendrogram (default is "level").
    p: int, the truncation parameter (e.g., number of levels for "level" mode).
    method: str, the linkage method to use (default is 'ward').
    cmap: str, the colormap for the heatmap (default is 'viridis').
    context_kws: dict, additional keyword arguments for seaborn's plotting context.
    show_log: bool, whether to apply a log transform to the matrix (default is False).
    cluster_colors: list of colors, custom colors for clusters (default is None).
    row_colorbar_size: tuple of float, size of the row colorbar (width, height) in figure units.
    row_colorbar_location: tuple of float, location of the row colorbar (x, y) in figure units.
    row_colorbar_tick_labels: list of str, custom tick labels for the row colorbar (default is None).
    col_colorbar_size: tuple of float, size of the column colorbar (width, height) in figure units.
    col_colorbar_location: tuple of float, location of the column colorbar (x, y) in figure units.
    col_colorbar_tick_labels: list of str, custom tick labels for the column colorbar (default is None).

    Returns:
    ----------
    clusters: numpy array, cluster assignments for each row.
    reordered_indices: numpy array, reordered row indices based on the dendrogram.
    g: clustermap object.
    """

    # Check if original_shape matches the number of columns
    num_columns = matrix.shape[1]

    if original_shape is not None:
        original_rows, original_cols = original_shape
        
    else:
        original_rows = original_cols = int(np.sqrt(num_columns).item())

    if original_rows * original_cols != num_columns:
        raise ValueError("original_shape does not match the number of columns in the matrix.")

    
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
        cluster_colors = sns.color_palette("Set3", num_clusters)

    row_colors = [cluster_colors[label - 1] for label in clusters]

    # Compute x and y coordinates for each column
    indices = np.arange(num_columns)
    x_coords = indices // original_cols  # Row index in original matrix
    y_coords = indices % original_cols   # Column index in original matrix


    # Generate discrete colormaps for x and y coordinates
    cmap_blue = ListedColormap(plt.cm.Blues(np.linspace(0.15, 0.8, original_rows+1)))
    cmap_grey = ListedColormap(plt.cm.Greys_r(np.linspace(0.15, 0.8, original_cols+1)))

    # Map coordinates to discrete colors
    
    y_colors = [ cmap_blue(i) for i in y_coords ]
    col_colors = y_colors #[ y_colors, x_colors ]

    # Prepare matrix for display
    # show_matrix = np.log(matrix + 1e-30) if show_log else matrix
    if show_log:
        matrix_raw = matrix.copy()
        matrix_log = np.log(matrix+1e-20)
        show_matrix = np.where(matrix_raw == 0, 0, matrix_log-matrix_log.min())
    else:
        show_matrix = matrix
    
    
    # Create clustermap with column colors
    with plt.rc_context(context_kws):
        g = sns.clustermap(
            show_matrix,
            row_linkage=row_linkage,
            col_linkage=None,
            col_cluster=False,
            cmap=cmap,
            row_colors=row_colors,
            col_colors=col_colors,  # Added column colors
            dendrogram_ratio=(0.2, 0),
            colors_ratio = (0.03, 0.02),
            figsize=(9, 8),
            vmin = show_matrix[show_matrix!=0].min(),
            cbar_kws=cbar_kws,
            cbar_pos = (0.95, 0.75, 0.04,0.2),
            rasterized=True,
        )
    
    
    # Customize heatmap appearance
    g.ax_heatmap.set_xticks([])
    g.ax_heatmap.set_yticks([])

    # Add legend for row clusters
    legend_patches = [
        Patch(color=cluster_colors[i], label=f"Cluster {i + 1}")
        for i in range(num_clusters)
    ]
    g.ax_heatmap.legend(
        handles = legend_patches,
        title = False,
        bbox_to_anchor = (1.02, 0.0, 0.35, 0.45),
        loc = 'lower left',
        borderaxespad = 0 ,
        fontsize=14,
        frameon=False,
    )
    
    hm_pos = g.ax_heatmap.get_position()
    col_pos = g.ax_col_colors.get_position()
    bottom_col_ax = g.fig.add_axes([hm_pos.x0, hm_pos.y0 - col_pos.height - 0.005 , hm_pos.width, col_pos.height  ])
    # bottom_col_ax.axis("off")
    x_colors = np.array([ [i] for i in x_coords ]).reshape(1,-1)
    sns.heatmap(x_colors, cmap=cmap_grey, ax=bottom_col_ax, cbar=False)
    bottom_col_ax.set_xticks(np.arange(5,105,10))
    bottom_col_ax.set_xticklabels(col_colorbar_tick_labels_y)

    # Add a continuous colorbar for column colors (x-coordinates, blue)
    cax_col_x = g.fig.add_axes([col_colorbar_location_x[0], col_colorbar_location_x[1], col_colorbar_size[0], col_colorbar_size[1]])
    cb_col_x = ColorbarBase(
        cax_col_x,
        cmap = cmap_blue,
        boundaries = np.arange(original_rows + 1)/original_rows - 0.05,
        ticks = np.arange(original_rows),
        orientation = 'vertical'
    )
    if col_colorbar_tick_labels_x is not None:
        cb_col_x.set_ticks(np.arange(original_rows)/original_rows)
        cb_col_x.set_ticklabels(col_colorbar_tick_labels_x)
    # cb_col_x.set_label('X Coordinate')

    # Add a continuous colorbar for column colors (y-coordinates, grey)
    # if col_colorbar_location_y is None:
    #     col_colorbar_location_y = col_colorbar_location_x
    #     col_colorbar_location_y[0] += 0.06
    # cax_col_y = g.fig.add_axes([col_colorbar_location_y[0], col_colorbar_location_y[1] , col_colorbar_size[0], col_colorbar_size[1]])
    # cb_col_y = ColorbarBase(
    #     cax_col_y,
    #     cmap=cmap_grey,
    #     boundaries = np.arange(original_cols + 1)/original_cols - 0.05,
    #     ticks = np.arange(original_cols),
    #     orientation = 'vertical'
    # )
    # if col_colorbar_tick_labels_y is not None:
    #     cb_col_y.set_ticks(np.arange(original_cols)/original_cols)
    #     cb_col_y.set_ticklabels(col_colorbar_tick_labels_y)
    # # cb_col_y.set_label('Y Coordinate')

    plt.show()
    
    return clusters, reordered_indices, g


# Plot scaled gene expression trends for TFgroup1 along pseudotime in the Neu branch

def plot_gene_trends(adata, gene_list, gene_trend_key, pseudotime_key='palantir_pseudotime', n_bins=100, PINN_params=None, para_name=None, ax=None):
    """
    Plot gene trends for a list of genes along pseudotime.
    If the PINN_params and para_name are provided, it will also plot the PINN parameters on a secondary y-axis.
    
    Args
    ----------
    adata: AnnData object containing the data.
    gene_list: List of genes to plot.
    gene_trend_key : the varm key in adata that contains the gene trends.
    pseudotime_key: Key in adata.obs that contains pseudotime values.
    n_bins: Number of bins for pseudotime.
    PINN_params : Optional, a numpy array of PINN parameters to plot on a secondary y-axis.
    para_name : Optional, a string to label the secondary y-axis for PINN_params.

    Returns
    ----------
    fig, ax, ax2

    Example
    ----------
    >>> TFgroup1 = ["Cebpe", "Clec4a2", "Cst7", "Elane", "Fcgr3", "Prtn3", "S100a8", "Wfdc21"]

    >>> pint_v_project = tl.aggregate_params_by_pseudotime(adata_neu, adata_neu.obsm['v_norm'].T, 
                            param_names='v',  pseudotime_key='palantir_pseudotime', nbins=100, return_y=True)

    >>> plot_gene_trends(adata_raw, TFgroup1, PINN_params=pint_v_project[3], para_name='PINN Day27 v')
    """
    # Ensure the gene list is in the correct format
    
    overlap_genes = set(gene_list).intersection(adata.var_names)
    print("gene not found :", set(gene_list) - overlap_genes)
    gene_trends_df = adata.varm[gene_trend_key].loc[list(overlap_genes)]

    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.gca()
    else:
        fig = ax.figure

    x = gene_trends_df.columns.astype(float)  # Convert column names to float for plotting
    # Plot gene trends
    for gene in gene_trends_df.index:
        ax.plot(x, gene_trends_df.loc[gene], label=gene)

    # Set x-ticks to only 5 evenly spaced ticks
    xticks = np.linspace(0, len(gene_trends_df.columns) - 1, 5, dtype=int)


    if PINN_params is not None and para_name is not None:
        # Plot pint_project[3] on a secondary y-axis
        ax2 = ax.twinx()
        # Align x for pint_project[3] (length 99) to gene_trends_df columns (length N)
        x_pint = np.linspace(0, x.max(), n_bins-1)

        ax2.plot(x_pint, PINN_params, color='black', linestyle='--', label=para_name)
        ax2.set_ylabel("PINT v_norm")
        lines2, labels2 = ax2.get_legend_handles_labels()
    else:
        ax2 = None
        lines2, labels2 = [], []

    # # Legends
    lines1, labels1 = ax.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.15, 1), loc='upper left')

    # ax.set_xticks(np.linspace(0,1, 5))
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("Scaled gene expression")

    # plt.title("Gene trends for Neutrophil genes along pseudotime (Neu branch)")
    plt.tight_layout()

    return fig, ax, ax2