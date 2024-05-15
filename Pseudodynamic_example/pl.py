import os
import numpy as np
import pandas as pd
import torch 
from torch import nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import callbacks 
from pytorch_lightning import loggers as pl_loggers

import models
from reader import Pdyn_ExtractDataset

# define neural network surrogate
u_theta = models.MLP_surrogate(channels = [2, 32, 32, 1], activation_fn='Tanh')

# pseudo dynamics model
checkpoint_path ="/home/wergillius/Project/PINN_dynamics/logs/Cspline_PINN/lightning_logs/version_4/checkpoints/epoch=999-total_loss=10.15423775.ckpt"
log_model = models.Cspline_PINN.load_from_checkpoint(checkpoint_path)
train_iter = iter(train_DL)
batch = next(train_iter)
s_col, t_col, s_all, t_b, u_b, Mean, Var = log_model.get_data(batch, False)

grid_s = np.linspace(0,1,300)

# predict
u_pred_b = log_model.u(s_all, t_b)

u_pred_b = u_pred_b.detach().numpy()
u_b = u_b.detach().numpy()[0]
N_theta = 0.5*(u_pred_b[:,1:]+u_pred_b[:,:-1]).sum(axis=1)
Mean = Mean.detach().numpy().flatten()
Var = Var.detach().numpy().flatten()
Mean
N_theta
# predict behavior
grid_ts = s_all[0,:,0]

v_knots  = log_model.v.y.detach().numpy()
g_knots  = log_model.g.y.detach().numpy()
D_knots  = log_model.D.y.detach().numpy()

v_curve  = log_model.v(grid_ts, 0).detach().numpy()
g_curve  = log_model.g(grid_ts, 0).detach().numpy()
D_curve  = log_model.D(grid_ts, 0).detach().numpy()
v_knots.shape, D_curve.shape

from matplotlib import pyplot as plt
from matplotlib import cm

fig, axs = plt.subplots(1, 3, figsize=(14,3), dpi=400)
axs = axs.flatten()

labels = ['v', 'g', 'D']
colors = cm.Set2([4,5,6])

for i,knots in enumerate([v_knots, g_knots, D_knots]):
    axs[i].scatter(np.linspace(0,1, len(knots)), knots, marker='o', label=labels[i], color=colors[i])

for i,behavior in enumerate([v_curve, g_curve, D_curve]):
    axs[i].plot(grid_s, behavior, label=labels[i], color=colors[i])
    axs[i].set_title(r"$"+labels[i]+r"$", fontsize=14)
    

# fig.suptitle('estimated havior parameters \n \n', fontsize=24)
axs[1].set_xlabel('cell state')
axs[0].set_ylabel('value')
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