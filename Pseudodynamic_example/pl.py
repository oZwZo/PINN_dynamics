import os
import numpy as np
import pandas as pd
import torch 
import models
from reader import Pdyn_ExtractDataset
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

def behavior_curves(log_model, n_grid=300):

    s = np.linspace(0, 1, n_grid)
    grid_ts = torch.from_numpy(s)

    v_knots  = log_model.v.y.detach().numpy()
    g_knots  = log_model.g.y.detach().numpy()
    D_knots  = log_model.D.y.detach().numpy()

    v_curve  = log_model.v(grid_ts, 0).detach().numpy()
    g_curve  = log_model.g(grid_ts, 0).detach().numpy()
    D_curve  = log_model.D(grid_ts, 0).detach().numpy()

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

    fig = plt.figure(figsize=(4,3), dpi=400)

    for i,t in enumerate(train_DS.T_b):
        plt.plot(np.linspace(0,1,300), u_pred_b[t], '--', label='day %s'%t)

    plt.title('prediction')

    plt.xlabel('cell state')
    plt.ylabel('density')
    plt.legend(ncol=2)
    fig, axs = plt.subplots(len(train_DS.T_b)//2, 2, figsize=(6,6), sharex=True, 
                        gridspec_kw={'hspace':0.4},  dpi=300)
    axs = axs.flatten()

    for i,t in enumerate(train_DS.T_b):
        axs[i].plot(np.linspace(0,1,300), u_b[t],  label='observed')
        axs[i].plot(np.linspace(0,1,300), u_pred_b[t], '--', label='pred')
        axs[i].set_title('day %s'%t)
        if i%2 ==0 :
            axs[i].set_ylabel("density")

    axs[0].legend()
    axs[4].set_xlabel('cell state')
    axs[5].set_xlabel('cell state')

    return fig, axs