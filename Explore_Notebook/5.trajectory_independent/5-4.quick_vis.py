import os, argparse, typing
os.chdir("/ssd/users/Wergillius/Project/PINN_dynamics")

import numpy as np
import pandas as pd
import scanpy as sc
import torch 
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from functools import partial
import pytorch_lightning as pl
from pytorch_lightning import callbacks 
from pytorch_lightning import loggers as pl_loggers

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import PINN
from PINN import models as models
from PINN import reader 
from PINN import functions as fns
from matplotlib import cm
from matplotlib.colors import Normalize

# CHANG THIS !!!!!
ckpt_path = "logs/tom_pos-DM_EigenVectors_multiBnch/MLP_full_tsense/lightning_logs/version_0/checkpoints/epoch=55-total_loss=0.00004008.ckpt"

# MODEL CLASS and define model
model_class = ckpt_path.split("/")[2].replace("_tsense","")
model_class = eval(f"models.{model_class}")
mlp_model = model_class.load_from_checkpoint(ckpt_path)
device = mlp_model.device

data_name = ckpt_path.split("/")[1].split("-")[0]
adata = sc.read_h5ad(f"data/{data_name}.h5ad")
timepoints = adata.uns['pop']['t']

n_timepoint = 7
n_dimension =10

# the DS
DS_t5 = reader.HigDim_AnnDS(AnnData=adata, n_timepoint=n_timepoint,  
                            n_dimension = n_dimension,
                              cellstate_key="DM_EigenVectors",  #'DM_EigenVector'
                              log_transform=False,
                              norm_time=False
                              )


# getting processed cell state and density from DataSet
t5_ad = DS_t5.adata.copy()
u_b = DS_t5.u_b.cpu().numpy().reshape(n_timepoint, -1)

s_all = DS_t5.s.float().requires_grad_()
t_b = DS_t5.t_b.float().requires_grad_()
cellstate = torch.from_numpy(DS_t5.cellstate).float().requires_grad_()


# density
u_pred = mlp_model.predict_param(DS_t5, param='u');
PINN.pl.params_in_umap(t5_ad, u_b, param='u', cell_of_t=False);
PINN.pl.params_in_umap(t5_ad, u_pred, param='u_pred', cell_of_t=False);
PINN.pl.params_in_umap(t5_ad, u_pred - u_b, param='u_error', cell_of_t=False);

# other params
dynamics_params = ['g', 'v', 'D']

if mlp_model.time_sensitive:
    for param in dynamics_params:
        param_pred = mlp_model.predict_param(DS_t5, param=param);
        if param_pred.shape[1] != t5_ad.shape[0]:
            param_pred = param_pred.reshape(5, -1, n_dimension)
            param_pred = mlp_model.trace_div(param_pred,)
        PINN.pl.params_in_umap(t5_ad, param_pred, param=param, cell_of_t=False);
else:
    for param in dynamics_params:
        t5_ad.obs[param] = mlp_model.predict_param(DS_t5, param=param);
    sc.pl.umap(t5_ad, color=dynamics_params, ncols=3, frameon=False, size=50)    



device = mlp_model.device
dudt, growth, drift, diffuse = mlp_model.equation(s_all.to(device), t_b.to(device))
# dudt, growth, drift, diffuse = mlp_model.equation(s_all.to(device), t_b.to(device))

# visualize residual
residual = dudt - (growth - drift + diffuse) 

# reshape
def pred_to_nday(x, n_timepoint=n_timepoint, n_dim=n_dimension): 
    """
    use to reshape prediction
    """
    nday = x.detach().cpu().numpy().reshape(n_timepoint,-1)

    if nday.shape[-1] != t5_ad.shape[0]:
        nday = nday.reshape(n_timepoint, -1, n_dim)
    return nday

residual = pred_to_nday(residual)
growth = pred_to_nday(growth)
dudt = pred_to_nday(dudt)
drift = pred_to_nday(drift)
diffuse = pred_to_nday(diffuse)

PINN.pl.params_in_umap(t5_ad, residual, param='residual', cell_of_t=False);
PINN.pl.params_in_umap(t5_ad, growth, param='growth', cell_of_t=False);
PINN.pl.params_in_umap(t5_ad, dudt, param='dudt', cell_of_t=False);
PINN.pl.params_in_umap(t5_ad, drift, param='drift', cell_of_t=False);

PINN.pl.params_in_umap(t5_ad, diffuse, param='diffuse', cell_of_t=False);




# animation
uprd_by_time = []
continuous_t = np.arange(timepoints[:n_timepoint].min(), timepoints[:n_timepoint].max()+1)
for t in continuous_t:
    t_ts = torch.full(size=(cellstate.shape[0],), fill_value=t).float().requires_grad_()
    u_pred = mlp_model.u(cellstate.to(device), t_ts.to(device))
    uprd_by_time.append(
        u_pred.detach().cpu().numpy()
    )
color_norm = Normalize(vmin=uprd_by_time[0].max(), vmax=uprd_by_time[-1].max())

sampled_t = continuous_t[::5]
sampled_i = [i for i,t in enumerate(continuous_t) if t in sampled_t]

fig, axs = plt.subplots(1, len(sampled_i), figsize=(len(sampled_i)*2.5,2))
axs = axs.flatten()
j = 0
coords = DS_t5.adata.obsm['X_umap']

for i,t in zip(sampled_i, sampled_t):
    axs[j].scatter(coords[:,0], coords[:,1], c=uprd_by_time[i], s=3, norm=color_norm, label=t)
    j+=1




fig , ax = plt.subplots(1,1, dpi=300)
coords = DS_t5.adata.obsm['X_umap']

def init():
    ax.scatter(coords[:,0], coords[:,1], color='lightgray', alpha=0.7, s=3)
    ax.set_title("Day %d"%t)

def run(data):
    if data>0:
        ax.clear()   
        # ax.scatter(coords[:,0], coords[:,1], color='lightgray', alpha=0.7,  size=3)
        t = continuous_t[data]
        upred_t = uprd_by_time[data]

        ax.scatter(coords[:,0], coords[:,1], c=upred_t, s=3, norm=None)
        
        ax.set_title("Day %d"%t)
    else:
        pass

ani = animation.FuncAnimation(fig, run, frames=len(continuous_t), interval=10, init_func=init)  # 製作動畫
ani.save(f"Explore_Notebook/5.trajectory_independent/{data_name}_u.gif", fps=5, writer='pillow') 
