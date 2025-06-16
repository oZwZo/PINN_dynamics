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
from PINN import reader, models, pl, tl
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from TorchDiffEqPack import odesolve
from torchdiffeq import odeint_adjoint as odeint

import matplotlib as mpl
import seaborn as sns
from matplotlib.patches import Patch

from matplotlib.backends.backend_pdf import PdfPages

os.chdir("/ssd/users/Wergillius/Project/PINN_dynamics")



def savefig(fig, plot_save, name):
    fig.savefig(f'{plot_save}/{name}.png', transparent=True, bbox_inches='tight', dpi=600)



# %% [markdown]
# # Load svaed checkpoint using Experiment config

# %%
config_dir = "logs/tompos_leavout_t5n7/pde_params"
config_files = [f"{config_dir}/V{i}_config.json" for i in range(0,4)]

config_files

# %%
DS_configs = []
ckpts = []
for config_file in config_files:
    configV = PINN.ExperimentConfig(config=config_file)
    DS_configs.append(configV.dataset_config)
    ckpts.append(configV.find_lastest_ckpt())

# %%
model0 = PINN.models.pde_params.load_from_checkpoint(ckpts[0])
model1 = PINN.models.pde_params.load_from_checkpoint(ckpts[1])
model2 = PINN.models.pde_params.load_from_checkpoint(ckpts[2])
model3 = PINN.models.pde_params.load_from_checkpoint(ckpts[3])

# %%
# the dataset config is the same among all versions
DS_configs

# %%
# so we can load 1 DS
adata = sc.read_h5ad("data/tom_pos.h5ad")

# %%
# set to full timepoints
DS_configs[0]['timepoint_idx'] = None
DS_full = PINN.reader.TwoTimpepoint_AnnDS(AnnData=adata, **DS_configs[0])
DM_range = DS_full.cellstate.max(axis=0, keepdims=True) - DS_full.cellstate.min(axis=0, keepdims=True)

if DS_configs[0]['norm_time'] != 'min_minus':
    day0 = adata.uns['pop']['t'][0]
else:
    day0 = 1


# %%
def get_norm(v):

    v_norm = v / DM_range[None,:, :]
    v_norm = np.sqrt(np.sum(v_norm**2, axis=2))

    return v_norm

params = {}

for i, model in enumerate([model0, model1, model2, model3]):

    g,v,D = model1.predict_param(train_DS=DS_full)
    g /= day0
    v /= day0
    D /= day0

    v_norm = get_norm(v)
    D_norm = get_norm(D)

    params["V"+str(i)] = g, v_norm, D_norm

# %%
params['V1'][1].shape

# %%
figg, axs = PINN.pl.params_in_umap(adata, params['V1'][0], param='g', cell_of_t=False, subplot_kws={'dpi':60});
figv, axs = PINN.pl.params_in_umap(adata, params['V1'][1], param=r'$|\vec{v}|$', cell_of_t=False, subplot_kws={'dpi':60});
figD, axs = PINN.pl.params_in_umap(adata, params['V1'][2], param=r'$|\vec{D}|$', cell_of_t=False, subplot_kws={'dpi':60});

# %%
plot_save = 'results/tompos_time_inv/pde_params/'

def savefig(fig, name):
    fig.savefig(f'{plot_save}/{name}.png', transparent=True, bbox_inches='tight', dpi=600)
    fig.savefig(f'{plot_save}/{name}.pdf', transparent=True, bbox_inches='tight', dpi=600)

plt.ioff()
figg, axs = PINN.pl.params_in_umap(adata, params[f'V{v}'][0], param='g', cell_of_t=False, subplot_kws={'dpi':600});
savefig(figg, f"V{v}_g")
figg.show(False)

figv, axs = PINN.pl.params_in_umap(adata, params[f'V{v}'][1], param=r'$|\vec{v}|$', cell_of_t=False, subplot_kws={'dpi':600});
savefig(figv, f"V{v}_vnorm")
figv.show(False)

figD, axs = PINN.pl.params_in_umap(adata, params[f'V{v}'][2], param=r'$|\vec{D}|$', cell_of_t=False, subplot_kws={'dpi':600});
savefig(figD, f"V{v}_Dnorm")
figD.show(False)

# %%
