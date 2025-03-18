# %load_ext autoreload
# %autoreload 2
import os,sys
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import PINN
from PINN import reader, models
from torchdiffeq import odeint
from TorchDiffEqPack.odesolver import odesolve
from torchdyn.core import NeuralODE
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm.auto import tqdm

os.chdir("/ssd/users/Wergillius/Project/PINN_dynamics")


# CHAGNE HERE !!
# model_ckpt = "logs/tom_pos-DM_scaled_n9/pde_params_tsense/lightning_logs/version_0/checkpoints/epoch=257-total_loss=0.76506275.ckpt"
model_ckpt = "logs/tom_pos-DM_scaled_n[0, 1, 2, 3, 4, 6, 8]/pde_params_tsense/lightning_logs/version_4/checkpoints/epoch=87-total_loss=1.34044445.ckpt"
device = 'cuda:4'


### 
# dataset input params
adata = sc.read_h5ad(f"data/tom_pos.h5ad")
TFs = pd.read_table("data/TF_list_mm.txt")['TF'].values

# sc.pp.subsample(adata, fraction=0.1)
cellstate_key = "DM_scaled"
n_dimension = 10
base_cellstate = adata.obsm[cellstate_key][:,:n_dimension].copy()

Dataset = reader.TwoTimpepoint_AnnDS(
                    AnnData=adata, 
                    timepoint_idx=9, 
                    n_dimension=10,
                    cellstate_key=cellstate_key,  #'DM_EigenVector'
                    log_transform=False,
                    norm_time=False,
                    deltax_key=None,
                    batchsize= 100)


##
# load model 
pde_model = models.pde_params.load_from_checkpoint(model_ckpt)
pde_model = pde_model.to(device)
DT = models.Density_Transfer(pde_model)


# initial cell

timepoint_tx_days = sorted(adata.obs.timepoint_tx_days.unique())
t0 = timepoint_tx_days[0]


def get_cb_of_celltype(adata, celltype, celltype_key='anno_man', n_interval=10):
    
    celltype_cbs = []
    for it,t in tqdm(enumerate(timepoint_tx_days[:-1])):
        # if it==0:
        #     torch.cuda.memory._record_memory_history()

        integrate_time = np.linspace(t/t0, timepoint_tx_days[it+1]/t0 ,n_interval+1) / pde_model.time_scale_factor

        # find cell of time
        start_cell = adata.obs.query(f"`{celltype_key}` == @celltype & `timepoint_tx_days` == @t").index
        start_cell = list(start_cell)
        celltype_cbs.append(start_cell)

    return celltype_cbs

def density_transfer_by_celltype(n_interval = 10, ncell = 300):
    
    Tmaps = {}
    Tmaps_norm = {}
    S_traj_lookup = {}
    celltype_cbs = []

    for it,t in tqdm(enumerate(timepoint_tx_days[:-1])):
        # if it==0:
        #     torch.cuda.memory._record_memory_history()

        integrate_time = np.linspace(t/t0, timepoint_tx_days[it+1]/t0 ,n_interval+1) / pde_model.time_scale_factor
        print(integrate_time)

        # find cell of time
        start_cell = adata.obs.query(f"`timepoint_tx_days` == @t").index
        start_cell = list(start_cell)
        celltype_cbs.append(start_cell)
        if len(start_cell) == 0:
            continue

        # define initital density and cellstates
        cell_index = [np.where(adata.obs_names == x)[0].item() for x in start_cell]
        u0 = Dataset.u_b[it, cell_index].float().to(device)
        s0 = torch.from_numpy(Dataset.cellstate[cell_index]).float().to(device)

        S_trajectory = DT.cellstate_drift(s0, integrate_time)
        Tmaps_t, Tmaps_t_norm = DT.transition_by_batch(s0, u0, integrate_time, n_interval=n_interval, ncell=ncell)

        del u0, s0
        
        S_traj_lookup[str(t)] = S_trajectory
        Tmaps_norm[str(t)] = Tmaps_t_norm
        Tmaps[str(t)] = Tmaps_t

    ## 
    # v4 , interval by pop t
    np.save(f"results/tompos_Tmap/Cellstate_traj_TM1_v4_by_time.npy", S_traj_lookup)
    np.save(f"results/tompos_Tmap/Tmaps_norm_TM1_v4_by_time.npy", Tmaps_norm)
    np.save(f"results/tompos_Tmap/Tmaps_TM1_v4_by_time.npy", Tmaps)

    return Tmaps, Tmaps_norm, celltype_cbs

if __name__ == '__main__':
    density_transfer_by_celltype()