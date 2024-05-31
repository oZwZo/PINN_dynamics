import os
import numpy as np
import pandas as pd
import torch 
from . import models
from .reader import Pdyn_ExtractDataset
from . import functions  as myfun
from matplotlib import pyplot as plt
from matplotlib import cm


def load_model():
    # define neural network surrogate
    u_theta = models.MLP_surrogate(channels = [2, 32, 32, 1], activation_fn='Tanh')

    # pseudo dynamics model
    checkpoint_path ="/home/wergillius/Project/PINN_dynamics/logs/Cspline_PINN/lightning_logs/version_4/checkpoints/epoch=999-total_loss=10.15423775.ckpt"
    log_model = models.Cspline_PINN.load_from_checkpoint(checkpoint_path)
    
    return log_model

# predict behavior

def behavior_curves(model, n_grid=300):

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


def density_by_time(u_b, u_pred_b, train_DS):

    fig, axs = plt.subplots(1,2,figsize=(8,3), dpi=300)
    cell_state = np.linspace(0,1,u_b.shape[1])
    
    if isinstance(u_b, torch.Tensor):
        u_b_ay = u_b.detach().numpy()
    if isinstance(u_pred_b, torch.Tensor):
        u_pred_b = u_pred_b.detach().numpy()
        
    for i in range(len(train_DS.T_b)):
        axs[0].plot(cell_state, u_b[i], label=i)
        axs[1].plot(cell_state, u_pred_b[i], '--', label=train_DS.T_b[i])
    
    axs[0].set_title('observation')
    axs[1].set_title('prediction')
    
    axs[0].set_xlabel('cell state')
    axs[1].set_xlabel('cell state')
    axs[0].set_ylabel('density')
    
    plt.legend(ncol=2)
    fig, axs = plt.subplots(int(np.ceil(len(train_DS.T_b)/2)), 2, figsize=(6,6), sharex=True, 
                        gridspec_kw={'hspace':0.4},  dpi=300)
    axs = axs.flatten()

    for i,t in enumerate(train_DS.T_b):
        axs[i].plot(cell_state, u_b[i],  label='observed')
        axs[i].plot(cell_state, u_pred_b[i], '--', label='pred')
        axs[i].set_title('day %s'%t)
        if i%2 ==0 :
            axs[i].set_ylabel("density")

    axs[0].legend()
    axs[-2].set_xlabel('cell state')
    axs[-1].set_xlabel('cell state')

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


def evaluate_behavior_for_cell(ad, model, dpt_key='dpt_pseudotime'):

    dpt = ad.obs[dpt_key].values
    scaled_dpt = myfun.scale_dpt(dpt)

    cellstate = torch.from_numpy(scaled_dpt)

    v_curve  = model.v(cellstate, 0).detach().numpy()
    g_curve  = model.g(cellstate, 0).detach().numpy()
    D_curve  = model.D(cellstate, 0).detach().numpy()

    ad.obs['PINN_v'] = v_curve
    ad.obs['PINN_g'] = g_curve
    ad.obs['PINN_D'] = D_curve

    # sc.pl.umap(ad, colors=['PINN_v','PINN_g', 'PINN_D'])

    return ad