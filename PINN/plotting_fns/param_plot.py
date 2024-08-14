# File:         PINN/plotting_fns/param_plot.py
# Usage:        PINN.pl.<fn_name>
# Description:  Visaulizing the dynamics behavior params for high dimensional modeling.
#               Some of the functions are shared between Trajectory-Dependent and Trajectory-Independent Modeling


import os
import numpy as np
import pandas as pd
import torch 
from .. import functions  as myfun
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from .density_plot import umap_by_time

# predict

def format_ay(array):
    formated  = [np.format_float_scientific(u, precision=2) for u in array]
    return formated

def params_in_umap(adata, prediction, timepoints=None, param='u', copy=True, cell_of_t=True):    
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
    
    for i, t in enumerate(timepoints):
        adata.obs[f'Day{t}_{param}'] = prediction[i]

    fig, axs = umap_by_time(lambda x: f'Day{x}_{param}', adata, timepoints, cell_of_t=cell_of_t)

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

