import os
import numpy as np
import pandas as pd
import torch
import random
import utility 
import scanpy as sc

from matplotlib import pyplot as plt
import seaborn as sns

timepoints = [ 3,   7,  12,  27,  49,  76, 112, 161, 269]

def umap_by_time(color_col, anndata, timepoints=timepoints):
    fig,axs = plt.subplots(1, 9, figsize=(30,2), dpi=100, gridspec_kw={'wspace':0.3})
    # axs = axs.flatten()
    axis_j = 0

    for t in timepoints:
        cbs = anndata.obs.query('`timepoint_tx_days` == @t').index

        sc.pl.umap(anndata, show=False, return_fig=False,  ax=axs[axis_j], alpha=0.5, s=50,frameon=False);
        sc.pl.umap(anndata[cbs], color=color_col, alpha=0.7, color_map='viridis', 
                return_fig=False,show=False, ax=axs[axis_j], s=50, frameon=False, 
                title='scaled pseudotime d%d'%t);
        
        axis_j += 1

    return fig, axs
def plot_along_pseudotime(color_col, anndata, pt_col='dpt_pseudotime', timepoints=timepoints):
    
    fig,axs = plt.subplots(2, 9, figsize=(30,4), dpi=100, gridspec_kw={'hspace':0.3})
    # axs = axs.flatten()
    axis_j = 0

    for t in timepoints:
        cbs = anndata.obs.query('`timepoint_tx_days` == @t').index

        sc.pl.umap(anndata, show=False, return_fig=False,  ax=axs[0,axis_j], alpha=0.5, s=50,frameon=False);
        sc.pl.umap(anndata[cbs], color=color_col, alpha=0.7, color_map='viridis', #colorbar_loc=None,
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
    fig,axs = plt.subplots(1, 9, figsize=(30,2), dpi=100, gridspec_kw={'wspace':0.3})
    # axs = axs.flatten()
    axis_j = 0

    for t in timepoints:
        cbs = anndata.obs.query('`timepoint_tx_days` == @t').index
        sns.scatterplot(data=anndata.obs.loc[cbs], x=pt_col, y = color_col, ax=axs[axis_j])
        axis_j += 1
    return fig