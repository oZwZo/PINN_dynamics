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

# CHANG THIS !!!!!
ckpt_path = "logs/tom_pos_atlas-DM_EigenVectors_multiBnch/MLP_woD_tsense/lightning_logs/version_1/checkpoints/epoch=394-total_loss=0.00001072.ckpt"


model_class = ckpt_path.split("/")[2].replace("_tsense","")
model_class = eval(f"models.{model_class}")
mlp_model = model_class.load_from_checkpoint(ckpt_path)
device = mlp_model.device

# adata = sc.read_h5ad("data/ery_mk.h5ad")
adata = sc.read_h5ad("data/tom_pos_atlas.h5ad")
timepoints = adata.uns['pop']['t']

n_timepoint = 5
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

# animation

cellstate = torch.from_numpy(DS_t5.cellstate).float().requires_grad_()

uprd_by_time = []
continuous_t = np.arange(timepoints.min(), timepoints.max()+1)
for t in continuous_t:
    t_ts = torch.full(size=(cellstate.shape[0],), fill_value=t).float().requires_grad_()
    u_pred = mlp_model.u(cellstate.to(device), t_ts.to(device))
    uprd_by_time.append(
        u_pred.detach().cpu().numpy()
    )

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

        ax.scatter(coords[:,0], coords[:,1], c=upred_t, s=3)
        
        ax.set_title("Day %d"%t)
    else:
        pass

ani = animation.FuncAnimation(fig, run, frames=len(continuous_t), interval=10, init_func=init)  # 製作動畫
ani.save("Explore_Notebook/5.trajectory_independent/tom+_u_change.gif", fps=5, writer='pillow') 



# density
u_pred = mlp_model.predict_param(DS_t5, param='u');
PINN.pl.params_in_umap(t5_ad, u_b, param='u', cell_of_t=True);
PINN.pl.params_in_umap(t5_ad, u_pred, param='u_pred', cell_of_t=True);
PINN.pl.params_in_umap(t5_ad, u_pred - u_b, param='u_error', cell_of_t=True);

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
dudt, growth, drift, diffuse = mlp_model.simplified_equation(s_all.to(device), t_b.to(device))
# dudt, growth, drift, diffuse = mlp_model.equation(s_all.to(device), t_b.to(device))

# visualize residual
residual = dudt - (growth - drift)

# reshape
residual = PINN.tl.pred_to_nday(residual)
growth = PINN.tl.pred_to_nday(growth)
dudt = PINN.tl.pred_to_nday(dudt)
drift = PINN.tl.pred_to_nday(drift)

PINN.pl.params_in_umap(t5_ad, residual, param='residual', cell_of_t=False);
PINN.pl.params_in_umap(t5_ad, residual, param='growth', cell_of_t=False);
PINN.pl.params_in_umap(t5_ad, dudt, param='dudt', cell_of_t=False);
PINN.pl.params_in_umap(t5_ad, dudt, param='drift', cell_of_t=False);


