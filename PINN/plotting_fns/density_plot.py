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

def umap_by_time(attribute, anndata, timepoints=timepoints, time_mask=True,  subplot_kws=None, cell_of_t=True, umap_kws=None):
    r"""
    A very basic functions plotting cellular attribute in the umap and stratified by time

    Arguments
    ----------
    attribute : str or callable, a function of time or a obs_key of the anndata
    anndata : anndata
    time_mask : bool or str, default to True. If bool, only visualize cells of each timepoint (True) or show all cells in each panel (False). If str, use the given obs key for timepoint selection.
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
        default_plotting_kw.update(subplot_kws)
        subplot_kws = default_plotting_kw
    fig,axs = plt.subplots(1, n_timepoints, **subplot_kws)
    # axs = axs.flatten()
    axis_j = 0
    
    default_umap_kws = {"alpha":0.7, "color_map":'viridis', "s":50}
    if umap_kws is None:
        umap_kws = default_umap_kws
    else:
        default_umap_kws.update(umap_kws)
        umap_kws = default_umap_kws

    if type(time_mask) == str:
        timepoint_key = time_mask
    else:
        timepoint_key = 'timepoint_tx_days' if 'timepoint_tx_days' in anndata.obs_keys() else 'timepoint'
    for t in timepoints:
        if len(timepoints) == 1:
            ax = axs
        else:
            ax = axs[axis_j]

        cbs = anndata.obs.query(f'`{timepoint_key}` == @t').index

        col = attribute(t) if isinstance(attribute, Callable) else attribute
        title = col if isinstance(attribute, Callable) else attribute+' d%d'%t

        sc.pl.umap(anndata, show=False, return_fig=False,  ax=ax, alpha=0.5, s=50,frameon=False);

        ad_t = anndata[cbs] if time_mask else anndata
        sc.pl.umap(ad_t, color=col,  
                return_fig=False,show=False, ax=ax, frameon=False, 
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


def obs_composition(adata, x_var, y_var, kind='bar', figkws={'figsize':[6,4]}, legend_kws={"bbox_to_anchor":(1.05, 1), "loc":'upper left'}, ax=None):

    # Calculate proportions
    cell_type_prop = (
        adata.obs
        .groupby(x_var)[y_var]
        .value_counts(normalize=True)
        .mul(100)
        .rename('percentage')
        .reset_index()
    )

    # Pivot for plotting
    prop_pivot = cell_type_prop.pivot(
        index=x_var,
        columns=y_var,
        values='percentage'
    ).fillna(0)

    # Get colors from Scanpy (if available)
    if f'{y_var}_colors' in adata.uns:
        colors = adata.uns[f'{y_var}_colors']
    else:
        colors = None

    # Create stacked area plot
    if ax is None:
        fig = plt.figure(**figkws)
        ax = plt.subplot()
    else:
        fig = ax.figure

    if kind=='area':
        prop_pivot.plot.area(
            ax=ax,
            stacked=True,
            color=colors,
            linewidth=0
        )

    if kind=='bar':
        prop_pivot.plot.bar(
        ax=ax,
        stacked=True,
        color=colors,
        width=0.65,       # Bar width
        edgecolor='white' # Optional: adds separation between bars
        )

    # Format plot
    ax.set_xlabel(x_var)
    ax.set_ylabel('Proportion (%)')
    # ax.set_title('Cell Type Composition Over Time')
    ax.legend(**legend_kws)
    # plt.tight_layout()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    # plt.show()

    return fig, ax

def celltype_proportion(p_celltype_melt, timepoints, cm_celltype, ct_key, density_key='value', x_lim=None,):

    """
    cell type proportion 
    """
    height = len(cm_celltype) * 0.35
    n_timepoint = len(timepoints)
    fig_subplot, axs = plt.subplots(1, n_timepoint, figsize=(3*n_timepoint, height), dpi=300, sharey=True)

    for i, t in enumerate(timepoints):
        t = str(t)
        ax=axs[i]

        sns.barplot(data=p_celltype_melt.query("`time` == @t"), 
                    # color=ct_key, #palette=cm_celltype,
                    edgecolor='gray', width=0.7,
                    x = density_key, y=ct_key, hue='data', ax=ax)
        
        for bars, hatch, legend_handle in zip(ax.containers, ['', '//'], ax.legend_.legendHandles):
            for bar, color in zip(bars, cm_celltype.values()):
                alpha = 1 if hatch == '' else 0.5
                bar.set_alpha(alpha)
                bar.set_facecolor(color)
                bar.set_hatch(hatch)
            # update the existing legend, use twice the hatching pattern to make it denser
            legend_handle.set_hatch(hatch + hatch)

        sns.despine() 
        axs[i].set_xlabel("")
        axs[i].set_title(t)
        if i!=0:
            axs[i].legend([], frameon=False)
        if x_lim is not None:
            axs[i].set_xlim(0, x_lim)

    axs[0].set_ylabel("")
    axs[n_timepoint//2].set_xlabel("cell type proportion")

    return fig_subplot


# continuous density transfer
def plot_tmap(tmap, log=False, ax=None, return_fig=False, **kws):

    # matrix shape sanity check
    if len(tmap.shape) == 1:
        squre_size = int(np.sqrt(tmap.shape[0]))
        tmap = tmap.reshape(squre_size,squre_size)
    
    elif (len(tmap.shape) == 2) and (tmap.shape[0] == 1):
        squre_size = int(np.sqrt(tmap.shape[1]))
        tmap = tmap.reshape(squre_size,squre_size)

    elif (len(tmap.shape) == 2) and (tmap.shape[0] == tmap.shape[1]):
        pass
    else:
        raise ValueError("the given transport matrix is not square")

    if ax is None:
        fig, ax = plt.subplots( figsize=(2,2) )
        
    if log:
        tmap =  np.log(tmap+1e-30)
    

    # set upper triangle to nan
    trui = np.triu_indices_from(tmap, k=1)
    tmap[trui] = np.nan
    
    # the actual plotting is here
    ax.matshow(tmap, **kws)
    ax.set_xticklabels('')
    ax.set_yticklabels('')    

    if return_fig and (ax is None):
        return fig, ax

    