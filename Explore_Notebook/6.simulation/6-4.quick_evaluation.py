import os, sys
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import PINN
from PINN import reader, models, pl, tl
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from TorchDiffEqPack import odesolve
from torchdiffeq import odeint

os.chdir("/ssd/users/Wergillius/Project/PINN_dynamics")

# CHANG THIS !!!!!
ckpt_path = "logs/TwoTimpepoint_u_dt_weight_n7/lightning_logs/version_0/checkpoints/epoch=28-total_loss=0.54973233 copy.ckpt"

model_name = ckpt_path.split("/")[1].replace("TwoTimpepoint_","").replace("_n7","")
model_class = eval(f"models.{model_name}")

# MODEL CLASS and define model
mlp_model = model_class.load_from_checkpoint(ckpt_path)
device = mlp_model.device

adata = sc.read_h5ad(f"data/tom_pos.h5ad")
timepoints = adata.uns['pop']['t']

cellstate_key = "DM_scaled"

n_timepoint = 7
n_dimension =10

# the DS
DS_t7 = reader.TwoTimpepoint_AnnDS(AnnData=adata, n_timepoint=n_timepoint,  
                            n_dimension = n_dimension,
                              cellstate_key=cellstate_key,  #'DM_EigenVector'
                              log_transform=False,
                              norm_time=False,
                              batchsize = 300
                              )
# adata
t7_ad = DS_t7.adata.copy()
u_b = DS_t7.u_b.cpu().numpy().reshape(n_timepoint, -1)
cellstate = torch.from_numpy(DS_t7.cellstate).float().requires_grad_()




##
## predict u with model forward
u_pred_ls = []
chunk_size= 500
t_list = timepoints / 15

s_ts = DS_t7.s.float().to(device)
t_ts = DS_t7.t_b.float().to(device)

with torch.no_grad():
    for i in tqdm(range(0, len(t_ts), chunk_size)):
        u_pred = mlp_model(s_ts[i:i+chunk_size],  t_ts[i:i+chunk_size])
        u_pred_ls.append(u_pred.detach().cpu().numpy())
u_pred_b = np.concatenate(u_pred_ls, axis=0).reshape(n_timepoint,-1)

PINN.pl.params_in_umap(t7_ad, u_b, param='u', cell_of_t=False);
PINN.pl.params_in_umap(t7_ad, u_pred_b, param='u pred by forward', cell_of_t=False);


##
## predict u with ode integrat
u_t_ls = []
chunk_size= 200
t_list = timepoints / 15

for i in tqdm(range(0, len(cellstate), chunk_size)):

    s0 = cellstate[i:i+chunk_size].to(device)
    tompos_u0 = torch.from_numpy(u_b[0, i:i+chunk_size]).float().to(device).requires_grad_()
    init_condition = (tompos_u0, s0)

    u_t, s_t = odeint(
                    mlp_model.ode_func,
                    init_condition,
                    torch.tensor(t_list).type(torch.float32).to(device),
                    atol=1e-5,
                    rtol=1e-5,
                    method='midpoint',
                    options = {'step_size': 0.1}
                )

    torch.cuda.empty_cache()

    u_t_ls.append(u_t.detach().cpu().numpy())
    del u_t, s_t
    torch.cuda.empty_cache()

u_int = np.concatenate(u_t_ls, axis=1).reshape(len(t_list), -1)

PINN.pl.params_in_umap(t7_ad, u_int[:-2], param='u by integrat', cell_of_t=False);
PINN.pl.params_in_umap(t7_ad, u_int, timepoints=timepoints, param='u by integrat', cell_of_t=False);
