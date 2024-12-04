# File:         PINN/plotting_fns/curve_plot.py
# Usage:        PINN.pl.<fn_name>
# Description:  Visualization functions mainly for single-trajecotry modeling
                

import os
import numpy as np
import pandas as pd
import torch 
from matplotlib import pyplot as plt
from matplotlib import cm


def behavior_curves(model, n_grid=300):
    r"""
    visualize the fit dynamic behavior curves for a single trajectory where
    the cellstate is 1-dimensional and range in (0,1)
    """

    s = np.linspace(0, 1, n_grid)
    grid_ts = torch.from_numpy(s)

    v_knots  = model.v.y.detach().numpy()
    g_knots  = model.g.y.detach().numpy()
    D_knots  = model.D.y.detach().numpy()

    v_curve  = model.v(grid_ts, 0).detach().numpy()
    g_curve  = model.g(grid_ts, 0).detach().numpy()
    D_curve  = model.D(grid_ts, 0).detach().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(14,3), dpi=400)
    axs = axs.flatten()

    labels = ['v', 'g', 'D']
    colors = cm.Set2([4,5,6])

    for i,knots in enumerate([v_knots, g_knots, D_knots]):
        axs[i].scatter(np.linspace(0,1, len(knots)), knots, marker='o', label=labels[i], color=colors[i])

    for i,behavior in enumerate([v_curve, g_curve, D_curve]):
        axs[i].plot(grid_ts, behavior, label=labels[i], color=colors[i])
        axs[i].set_title(r"$"+labels[i]+r"$", fontsize=14)
        

    # fig.suptitle('estimated havior parameters \n \n', fontsize=24)
    axs[1].set_xlabel('cell state')
    axs[0].set_ylabel('value')

    return fig, axs


def density_by_time(u_b, u_pred_b, T_b, xlabel='cell state'):
    r"""
    plot density curve by cellstate

    Augments:
    ---------
    u_b : [tensor, ndarray] : the observed density of shape [n_timepoints, n_grid]
    u_pred_b : [tensor, ndarray] : the predicted density of shape [n_timepoints, n_grid]
    T_b : a list of real-time 
    xlabel : the x label
    """
    fig, axs = plt.subplots(1,2,figsize=(8,3), dpi=300)
    cell_state = np.linspace(0,1,u_b.shape[1])
    
    if isinstance(u_b, torch.Tensor):
        u_b_ay = u_b.detach().numpy()
    if isinstance(u_pred_b, torch.Tensor):
        u_pred_b = u_pred_b.detach().numpy()
        
    for i in range(len(T_b)):
        axs[0].plot(cell_state, u_b[i], label=i)
        axs[1].plot(cell_state, u_pred_b[i], '--', label=T_b[i])
    
    axs[0].set_title('observation')
    axs[1].set_title('prediction')
    
    axs[0].set_xlabel(xlabel)
    axs[1].set_xlabel(xlabel)
    axs[0].set_ylabel('density')
    
    plt.legend(ncol=2)

    n_col = int(np.ceil(len(T_b)/2))
    fig, axs = plt.subplots(2 ,n_col, figsize=(n_col*4,6), sharex=True, 
                        gridspec_kw={'hspace':0.4},  dpi=200)
    axs = axs.flatten()

    for i,t in enumerate(T_b):
        axs[i].plot(cell_state, u_b[i],  label='observed')
        axs[i].plot(cell_state, u_pred_b[i],  label='pred')
        axs[i].set_title('day %s'%t)
        if i%n_col ==0 :
            axs[i].set_ylabel("density")
        
        axs[i].set_xlabel(xlabel)

    axs[0].legend()

    return fig, axs

def meshgrid_density_by_time(ub, ub_pred, train_DS, cellstate_1='cellstate1', cellstate_2='cellstate2', fill=True, fig_kws=None):
    r"""
    Visualize the observed and predicted 2D mesh grid density

    Augments:
    ---------
    ub : [array, tensor], the observed density
    ub_pred : array, tensor], density predicted by `u_theta`
    train_DS : [Dataset], the Training Dataset defined in PINN.reader, 
    cellstate_[1/2] : the label of the axis
    fig_kws : dict|None , the keyword augments to control the layout and other params of the subplots

    Returns
    ---------
    fig, axs : the matplotlib figure and subplot-axis

    Example
    ---------
    >>> ad = ery_mk_ad = sc.read_h5ad("<XXX>.h5ad")
    >>> train_DS = PINN.reader.MeshGrid_AnnDS(*args, **kwargs)
    >>> model = PINN.models.Cspline_PINN.load_from_checkpoint(*args, **kwargs)

    """
    # getting experimental info from training Dataset
    n_grid = train_DS.n_grid
    ndays = len(train_DS.popD['t'])

    # getting cell state coordinates
    if train_DS.s.shape[0] == train_DS.t_b.shape[0]:
        s = train_DS.s[:n_grid*n_grid]   
    else:
        s = train_DS.s   
    XX = s[:,0].reshape(n_grid,n_grid)    # in DS, meshgrid is flatten
    YY = s[:,1].reshape(n_grid,n_grid)

    # type check
    if isinstance(ub, torch.Tensor):
        ub = ub.detach().numpy()
    if isinstance(ub_pred, torch.Tensor):
        ub_pred = ub_pred.detach().numpy()

    
    # subplot panels
    fig_kws_base = { "figsize":(3*ndays,5),
                    "gridspec_kw":{'wspace':0.4, 'hspace':0.4}}  
    if fig_kws is None:
        fig_kws =  fig_kws_base
    else:
        fig_kws_base.update(fig_kws)
        fig_kws = fig_kws_base
        
    fig, axs = plt.subplots(2, ndays,  **fig_kws)

    for i, T in enumerate(train_DS.popD['t']):
        
        # XX and YY is the output of meshgird
        # Z is of shape 50, 50
        Z_ub = ub[i].reshape(n_grid, n_grid)
        if fill:
            axs[0,i].contourf(XX,YY, Z_ub, cmap='Blues')
        else:
            axs[0,i].contour(XX,YY, Z_ub, cmap='Blues')

        Z_pred = ub_pred[i].reshape(n_grid, n_grid)
        if fill:
            axs[1,i].contourf(XX,YY, Z_pred, cmap='Blues')
        else:
            axs[1,i].contour(XX,YY, Z_pred, cmap='Blues')


        # set title
        axs[0,i].set_title("Day %d\n\nobserved density" %T)
        axs[1,i].set_title("predicted density")

        # set x and y label
        # axs[0,i].set_xlabel(cellstate_1)
        axs[0,i].set_ylabel(cellstate_2)
        axs[1,i].set_xlabel(cellstate_1)
        axs[1,i].set_ylabel(cellstate_2)  
    
    return fig, axs


def predict_and_vis(model, data_batch, train_DS, curveplot=True, densityplot=True, return_pred=True):
    s_col, t_col, s_all, t_b, u_b, Mean, Var = model.get_data(data_batch, False)

    grid_s = np.linspace(0,1,s_all.shape[1])

    # predict
    u_pred_b = model.u(s_all, t_b)

    u_pred_b = u_pred_b.detach().numpy()
    u_b = u_b.detach().numpy()
    N_theta = 0.5*(u_pred_b[:,1:]+u_pred_b[:,:-1]).sum(axis=1)
    Mean = Mean.detach().numpy().flatten()
    Var = Var.detach().numpy().flatten()

    if curveplot:
        behavior_curves(model);
    if densityplot:
        density_by_time(u_b, u_pred_b, train_DS);
    
    print(N_theta)
    print(Mean)


    if return_pred:
        return u_pred_b, N_theta


# def evaluate_behavior_for_cell(ad, model, dpt_key='dpt_pseudotime'):

#     dpt = ad.obs[dpt_key].values
#     scaled_dpt = myfun.scale_dpt(dpt)

#     cellstate = torch.from_numpy(scaled_dpt)

#     v_curve  = model.v(cellstate, 0).detach().numpy()
#     g_curve  = model.g(cellstate, 0).detach().numpy()
#     D_curve  = model.D(cellstate, 0).detach().numpy()

#     ad.obs['PINN_v'] = v_curve
#     ad.obs['PINN_g'] = g_curve
#     ad.obs['PINN_D'] = D_curve

#     # sc.pl.umap(ad, colors=['PINN_v','PINN_g', 'PINN_D'])

#     return ad