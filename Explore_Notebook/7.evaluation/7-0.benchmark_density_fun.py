# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: PINN_torch
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2
import os, sys, re
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import time
import PINN
from functools import partial
from PINN import reader, models, pl, tl
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from TorchDiffEqPack import odesolve
from torchdiffeq import odeint_adjoint as odeint
import mellon
import matplotlib as mpl
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import pearsonr, spearmanr
from matplotlib.backends.backend_pdf import PdfPages

# %%
main_dir="/ssd/users/Wergillius/Project/PINN_dynamics"
os.chdir(main_dir)
TM1=[0,1,2,3,4,6,8]
TM2=[0,1,2,3,5]

data_name = 'tom_pos'
cellstate_key = 'DM_scaled'
timepoint_key = 'timepoint_tx_days'
n_dimension = 30 
ct_key = 'anno_man'

# %%
adata = sc.read_h5ad(f"data/{data_name}.h5ad")

# %%
gas = sc.read_h5ad(f"data/gastrulation.h5ad")

# %%
gas.obs['timepoint_tx_days'].unique()

# %%
gas

# %%
adata.obs[timepoint_key].unique()

# %%
weinreb = sc.read_h5ad(f"data/klein_subset.h5ad")

# %%
device='cuda:2'

# %% [markdown]
# # eval density func

# %%
sc.tl.leiden(adata, resolution=4, key_added='rs4.0')
adata.obs['rs4.0'].nunique()

# %%
u_theta  = PINN.models.pde_params.load_from_checkpoint("logs/tom_pos-DM_scaled_n9/pde_params_tsense/lightning_logs/version_0/checkpoints/epoch=224-total_loss=0.81510490.ckpt", map_location='cpu')
u_theta = u_theta.to(device)

# %%
X = adata.obsm[cellstate_key]
cs = torch.from_numpy(X[:,:10]).to(device).float()

# %%
log_u_t = []
t0 = adata.uns['pop']['t'][0]
for t in adata.uns['pop']['t']:
    t_in = torch.full((X.shape[0],), t/t0).to(device).float()
    with torch.no_grad():
        u = u_theta.u(cs, t_in)
        log_u_t.append(u.cpu().numpy())
        del u

log_u_t = np.stack(log_u_t)

log_u_t.shape

# %%
np.save(f"data/tom_pos_surrogate_logu.npy", log_u_t)

# %%

# %%
# mellon
timepoints = adata.uns['pop']['t']
t_pred = mellon.Predictor.from_json(f"data/tom_pos_mellon_timecontinuous_predictor.json")
u_time_model = np.stack([t_pred(X, np.full((X.shape[0],), np.log(t))) for t in timepoints])

# %%
np.exp(log_u_t)

# %%
np.exp(u_time_model)

# %%
adata

# %%
p_metacell = []
for t in timepoints:
    df_t = adata.obs.query('`timepoint_tx_days` == @t')[['rs50', 'cellid']]
    n_cell = df_t.shape[0]
    r50_count_t = df_t.groupby('rs50').agg('count') / n_cell
    p_metacell.append(r50_count_t.values.flatten())

# %%
p_metacell = np.stack(p_metacell)

# %%
p_metacell.shape

# %%
adata.obsm['TIGON_u'] = np.load("/home/wergillius/Project/PINN_dynamics/data/tom_pos_TIGON_u_sigma_0p5.npy").T
adata.obsm['mellon_u'] = np.exp(u_time_model).T
adata.obsm['surrogate_u'] = np.exp(log_u_t).T

# %%

# %%
T_sum = adata.obsm['TIGON_u'].sum(axis=0) * 1e5
T_sum = T_sum/T_sum[0]

m_sum = adata.obsm['mellon_u'].sum(axis=0) /1e7
m_sum = m_sum/m_sum[0]

s_sum = adata.obsm['surrogate_u'].sum(axis=0)
s_sum = s_sum/s_sum[0]

# %%
import matplotlib.pyplot as plt

# %%
fig, ax = plt.subplots(1,1, figsize=(5,3), dpi=300)
ax.plot(range(9), T_sum, marker='o', label='TIGON')
ax.plot(range(9), m_sum, marker='o', label='mellon')
ax.plot(range(9), s_sum, marker='o', label='u_theta')

ax.set_xticklabels(timepoints)
ax.set_ylabel("total density sum")
ax.set_xlabel("timepoints")
ax.legend()

# %%
bulk_surrogate_u = PINN.tl.get_pseudobulk(adata, 'surrogate_u', pseudobulk_key='rs50').T
bulk_surrogate_u = bulk_surrogate_u / bulk_surrogate_u.sum(axis=1).reshape(-1,1)

bulk_TIGON_u = PINN.tl.get_pseudobulk(adata, 'TIGON_u', pseudobulk_key='rs50').T
bulk_TIGON_u = bulk_TIGON_u / bulk_TIGON_u.sum(axis=1).reshape(-1,1)

bulk_mellon_u = PINN.tl.get_pseudobulk(adata, 'mellon_u', pseudobulk_key='rs50').T
bulk_mellon_u = bulk_mellon_u / bulk_mellon_u.sum(axis=1).reshape(-1,1)

# %%
adata.write_h5ad(f"data/{data_name}.h5ad")

# %%
from scipy.special import kl_div
from scipy.stats import pearsonr
from scipy.stats import spearmanr

# %%
kld_s = [kl_div(p_metacell[i], bulk_surrogate_u[i]).sum() for i in range(9)]
kld_m = [kl_div(p_metacell[i], bulk_mellon_u[i]).sum() for i in range(9)]
kld_t = [kl_div(p_metacell[i], bulk_TIGON_u[i]).sum() for i in range(9)]

# %%
kld_df = pd.DataFrame({"u_theta":kld_s, "mellon":kld_m, "TIGON":kld_t})
kld_df['timepoints'] = timepoints

# %%
kld_df

# %%
import seaborn as sns

# %%
fig, ax = plt.subplots(1,1, figsize=(5,3), dpi=300)
ax.plot(range(9), kld_t, marker='o', label='TIGON')
ax.plot(range(9), kld_m, marker='o', label='mellon')
ax.plot(range(9), kld_s, marker='o', label='u_theta')

ax.set_xticklabels(timepoints)
ax.set_ylabel("KLD (density vs true)")
ax.set_xlabel("timepoints")
ax.legend()

# %%
bulk_umap = PINN.tl.get_pseudobulk(adata, 'X_umap', pseudobulk_key='rs50')

# %%
fig, axs = plt.subplots(4, 9, figsize=(3*9+4,13), dpi=300)
for i in range(9):
    axs[0, i].scatter( bulk_umap[:,0], bulk_umap[:,1] , c = bulk_TIGON_u[i], s=10, alpha=0.3, label='TIGON')
    axs[1, i].scatter( bulk_umap[:,0], bulk_umap[:,1] , c = bulk_mellon_u[i], s=10, alpha=0.3, label='mellon')
    axs[2, i].scatter( bulk_umap[:,0], bulk_umap[:,1] , c = bulk_surrogate_u[i], s=10, alpha=0.3, label='u_theta')
    axs[3, i].scatter( bulk_umap[:,0], bulk_umap[:,1] , c = p_metacell[i], s=10, alpha=0.3, label='true')
    

# %%
p_metacell[0]

# %% [markdown]
# # Weinreb

# %%
device

# %%
from denmarf import DensityEstimate

de = DensityEstimate.from_file("results/denmarf_model/tom_pos_DM_scaled_scaled_161_new_bound.pkl", device='cuda:2')

# %%
X = adata.obsm[cellstate_key]
X_scaled = X / X.std(axis=0).reshape(1,-1)

# %%
X.std(axis=0).reshape(1,-1)

# %%
log_density = de.score_samples(X_scaled)

# %%
log_density

# %%
np.isinf(log_density).sum()

# %%
np.quantile(log_density[~np.isinf(log_density)],[0,0.01,0.1,0.5,0.9,0.99,0.999])

# %%
log_density_clip = np.clip(log_density,a_min=-10, a_max=None)

# %%
log_density_clip

# %%
adata.obs['log_density_clip'] = log_density_clip

# %%
# %matplotlib inline

# %%
sc.pl.umap(adata, color='log_density_clip')

# %%
density = np.exp(log_density_clip)

# %%
density = density / density.sum()

# %%
np.quantile(np.log(density),[0,0.01,0.1,0.5,0.9,0.99,0.999])

# %%
gde = DensityEstimate.from_file("results/denmarf_model/gastrulation_DM_EigenVectors_multiscaled_6.5_new_bound.pkl", device='cuda:2')

# %%
X_gas = gas.obsm['DM_EigenVectors_multiscaled']

# %%
log_u_gas = gde.score_samples(X_gas)

# %%
np.isinf(log_u_gas).sum()

# %%
# new
np.quantile(log_u_gas, [0,0.01,0.1,0.5, 0.8 ,0.9,0.99,0.999])

# %%
# old
np.quantile(log_u_gas, [0,0.01,0.1,0.5, 0.8 ,0.9,0.99,0.999])

# %% [markdown]
#
# array([-2.01295937e+08, -1.11949264e+08, -9.55712179e+04, -9.11171622e+02,
#         9.02808920e+01,  1.63831099e+02,  1.94318236e+02,  2.07038313e+02])

# %%
gas.obs['log_density'] = np.clip(log_u_gas, a_min=-180, a_max=None)

# %%
sc.pl.umap(gas, color='log_density')

# %%
# nobound

# %%
gde_nob = DensityEstimate.from_file("results/denmarf_model/gastrulation_DM_EigenVectors_multiscaled_8.5.pkl", device='cuda:2')

# %%
u_nobound = gde_nob.score_samples(X_gas)

# %%
u_nobound.min()

# %%
np.quantile(u_nobound, [0,0.01,0.1,0.5,0.9, 0.99, 0.999])

# %%
gas.obs['u_nobound'] = np.clip(u_nobound, a_min=-200, a_max=None)

# %%
sc.pl.umap(gas, color='u_nobound')

# %%
