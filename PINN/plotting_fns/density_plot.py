import os
import numpy as np
import pandas as pd
import torch
import random
# import utility 
import scanpy as sc
from typing import Callable

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import matplotlib.animation as animation

timepoints = [ 3,   7,  12,  27,  49,  76, 112, 161, 269]

def umap_by_time(attribute, anndata, timepoints=timepoints, cell_of_t=True, subplot_kws=None, umap_kws={"alpha":0.7, "color_map":'viridis'}):
    r"""
    A very basic functions plotting cellular attribute in the umap and stratified by time

    Arguments
    ----------
    attribute : str or callable, a function of time or a obs_key of the anndata
    anndata : anndata
    cell_of_t : bool, default to True, only visualize cells of each timepoints. 
                If set to False, all cells will be shown in each panels.
    timepoints : iterable, list of real-time , like the number of columns

    Return
    ----------
    matplotlb figure and axes

    Example
    ----------
    >>> u_b = DataSet.u_b
    >>> adata = DataSet.adata
    >>> for i, t in enumerate(DataSet.popD['t']):
            adata.obs[f'Day{t}_u'] = u_b[i]
    >>> # use a lambda function as attributes to plot
    >>> PINN.pl.umap_by_time(lambda x: f'Day{x}_u', adata, DataSet.popD['t']);
    """

    n_timepoints = len(timepoints)

    default_plotting_kw = dict(figsize=(n_timepoints*2.7,2), gridspec_kw={'wspace':0.4})
    if subplot_kws is None:
        subplot_kws = default_plotting_kw
    else:
        subplot_kws = default_plotting_kw.update(subplot_kws)
    fig,axs = plt.subplots(1, n_timepoints, **subplot_kws)
    # axs = axs.flatten()
    axis_j = 0

    timepoint_key = 'timepoint_tx_days' if 'timepoint_tx_days' in anndata.obs_keys() else 'timepoint'
    for t in timepoints:
        cbs = anndata.obs.query(f'`{timepoint_key}` == @t').index

        col = attribute(t) if isinstance(attribute, Callable) else attribute
        title = col if isinstance(attribute, Callable) else attribute+' d%d'%t

        sc.pl.umap(anndata, show=False, return_fig=False,  ax=axs[axis_j], alpha=0.5, s=50,frameon=False);

        ad_t = anndata[cbs] if cell_of_t else anndata
        sc.pl.umap(ad_t, color=col,  
                return_fig=False,show=False, ax=axs[axis_j], frameon=False, 
                title=title, **umap_kws);
        
        axis_j += 1

    return fig, axs


def plot_along_pseudotime(color_col, anndata, pt_col='dpt_pseudotime', timepoints=timepoints):
    
    n_timepoints = len(timepoints)
    fig,axs = plt.subplots(2, n_timepoints, figsize=(n_timepoints*2.7,5), dpi=100, gridspec_kw={'wspace':0.4})
    # axs = axs.flatten()
    axis_j = 0

    for t in timepoints:
        cbs = anndata.obs.query('`timepoint_tx_days` == @t').index

        col = color_col(t) if isinstance(color_col, Callable) else color_col

        sc.pl.umap(anndata, show=False, return_fig=False,  ax=axs[0,axis_j], alpha=0.5, s=50,frameon=False);
        sc.pl.umap(anndata[cbs], color=col, alpha=0.7, color_map='viridis', #colorbar_loc=None,
                return_fig=False,show=False, ax=axs[0, axis_j], s=50, frameon=False, 
                title='scaled pseudotime d%d'%t);


        axs[0,axis_j].invert_xaxis()

        sns.kdeplot(anndata.obs.query('`timepoint_tx_days` == @t')[pt_col], label="d%d"%t, ax=axs[1,axis_j])
        sns.despine(ax=axs[1,axis_j])
        axs[1,axis_j].set_xlim(0,1)
        axs[1,axis_j].set_ylabel("")

        axis_j += 1

    axs[1,0].set_ylabel("density")
    axs[0,0].set_ylabel("UMAP-2")
    axs[0,0].set_xlabel("UMAP-1")
    fig.show()
    # axs[-1].axis("off")
    return fig

def scatter_density(color_col, anndata, pt_col='dpt_pseudotime', timepoints=timepoints):

    n_timepoints = len(timepoints)
    fig,axs = plt.subplots(1, n_timepoints, figsize=(n_timepoints*2.7,2), dpi=100, gridspec_kw={'wspace':0.4})
    # axs = axs.flatten()
    axis_j = 0

    for t in timepoints:
        cbs = anndata.obs.query('`timepoint_tx_days` == @t').index
        sns.scatterplot(data=anndata.obs.loc[cbs], x=pt_col, y = color_col, ax=axs[axis_j])
        axis_j += 1
    return fig

def resampling_animation_meshgrid(s, train_DS, save_path):
    """
    s : cellstate coordinates, [n_grid**2, 2]
    train_DS : `reader.MeshGrid_logDS` classs
    """
    fig , ax = plt.subplots(1,1, dpi=300)
    ax.set_xlim(0,1)         
    ax.set_ylim(0,1)         

    def init():
        ax.scatter(s[:,0], s[:,1], color='lightgray', s=3)

    def run(data):
        if data>0:
            ax.clear()
            reindex_i = train_DS.resampling_by_density(50, p=train_DS.density_P)
            ax.scatter(s[:,0], s[:,1], color='lightgray', s=3)
            ax.scatter(s[reindex_i,0], s[reindex_i,1], color='navy', s=6, marker='s')
        else:
            pass

    ani = animation.FuncAnimation(fig, run, frames=100, interval=10, init_func=init)  # make animation
    ani.save(save_path, fps=5, writer='pillow') 


def resampling_animation_umap(train_DS, save_path):
    """
    train_DS : `reader.HigDim_AnnDS` classs
    save_path : str, a path ends with .gif
    """
    coords = train_DS.adata.obsm['X_umap']
    resampling_rate = train_DS.resampling_rate

    fig , ax = plt.subplots(1,1, dpi=100) 

    def init():
        ax.scatter(coords[:,0], coords[:,1], color='lightgray', s=3)

    def run(data):
        if data>0:
            ax.clear()
            reindex_i = []
            for i in np.random.randint(low=0, high=len(train_DS), size=100):
                if np.random.random() <= resampling_rate:
                    i = train_DS.resampling_by_density(1, p=train_DS.density_P)
                    reindex_i.append(i.item())
                else:
                    reindex_i.append(i)
            reindex_i = np.array(reindex_i) % coords.shape[0]

            ax.scatter(coords[:,0], coords[:,1], color='lightgray', s=3)
            ax.scatter(coords[reindex_i,0], coords[reindex_i,1], color='navy', s=6, marker='s')
            ax.set_title("resampling rate = %f" %resampling_rate)
        else:
            pass

    ani = animation.FuncAnimation(fig, run, frames=200, interval=10, init_func=init)  # make animation
    ani.save(save_path, fps=5, writer='pillow') 


def stack_catplot(x, y, cat, stack, data, palette=sns.color_palette('Reds')):
    ax = plt.gca()
    # pivot the data based on categories and stacks
    # df = data.pivot_table(values=y, index=[cat, x], columns=stack, 
    #                       dropna=False, aggfunc='sum').fillna(0)
    ncat = data[cat].nunique()
    nx = data[x].nunique()
    nstack = data[stack].nunique()
    range_x = np.arange(nx)
    width = 0.8 / ncat # width of each bar
    
    for i, c in enumerate(data[cat].unique()):
        # iterate over categories, i.e., Conditions
        # calculate the location of each bar
        loc_x = (0.5 + i - ncat / 2) * width + range_x
        bottom = 0

        for j, s in enumerate(data[stack].unique()):
            # iterate over stacks, i.e., Hosts
            # obtain the height of each stack of a bar
            height_df = data.query(f"`{cat}` == @c & `{stack}`==@s")
            height_df = height_df.set_index(x)
            height = height_df.loc[data[x].unique(), y]
            # plot the bar, you can customize the color yourself
            
            hatch = '/' if i == 1 else None
            barcontainer = ax.bar(x=loc_x, height=height, 
                                  bottom=bottom, width=width*0.7, 
                                    color=palette[s], 
                                    # zorder=10, 
                                    lw=0.1,
                                    hatch=hatch, label=f"{c}: {s}")
            
  
            for bc in barcontainer:
                bc._hatch_color = mpl.colors.to_rgba("w")
                bc.stale = True
            
            # change the bottom attribute to achieve a stacked barplot
            bottom += height

    # make xlabel
    ax.set_xticks(range_x)
    ax.set_xticklabels(data[x].unique(), rotation=45)
    ax.set_ylabel(y)
    # make legend
    plt.legend(
            #     [Patch(facecolor=palette[i]) for i in range(ncat * nstack)], 
            #    [f"{c}: {s}" for c in data[cat].unique() for s in data[stack].unique()],
               bbox_to_anchor=(1.05, 0.8), loc='upper left', borderaxespad=0., ncol=2)
    plt.grid()
    return ax