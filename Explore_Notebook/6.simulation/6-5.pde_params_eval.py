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
pde_model = model_class.load_from_checkpoint(ckpt_path)
device = pde_model.device

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
## predict u with ode integrat
u_int_all = []
chunk_size= 500
# t_list = timepoints / 15
it = 0
for t0, t1 in zip(timepoints[:-1], timepoints[1:]):

    u_t_ls = []

    for i in tqdm(range(0, len(cellstate), chunk_size)):

        s0 = cellstate[i:i+chunk_size].to(device)
        tompos_u0 = torch.from_numpy(u_b[it, i:i+chunk_size]).float().to(device).requires_grad_()
        init_condition = (tompos_u0, s0)

        step_size = np.around((t1 - t0)/15, decimals=1).item() 
        step_size = step_size if step_size > 0 else 0.05
        step_size = min(step_size, 0.4)


        u_t, s_t = odeint(
                        pde_model.ode_func,
                        init_condition,
                        torch.tensor([t0, t1]).type(torch.float32).to(device),
                        atol=1e-8,
                        rtol=1e-8,
                        method='midpoint',
                        options = {'step_size': step_size}
                    )

        torch.cuda.empty_cache()

        u_t_ls.append(u_t.detach().cpu().numpy())
        del u_t, s_t
        torch.cuda.empty_cache()

    u_int = np.concatenate(u_t_ls, axis=1)
    
    u_int_all.append(u_int)
    it+=0

PINN.pl.params_in_umap(t7_ad, u_int[:-2], param='u by integrat', cell_of_t=False);
PINN.pl.params_in_umap(t7_ad, u_int, timepoints=timepoints, param='u by integrat', cell_of_t=False);
