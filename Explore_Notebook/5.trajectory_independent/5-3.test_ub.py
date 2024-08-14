import os, argparse, typing
os.chdir("/ssd/users/Wergillius/Project/PINN_dynamics")

import numpy as np
import pandas as pd
import scanpy as sc

import matplotlib.pyplot as plt

import PINN
from PINN import models as models
import matplotlib.animation as animation
from PINN import reader 


# adata = sc.read_h5ad("data/ery_mk.h5ad")
adata = sc.read_h5ad("data/tom_pos_atlas.h5ad")
timepoints = adata.uns['pop']['t']

n_dimension = 5
log_transform = False
resampling_rate = 0

processing = 'log-norm' if log_transform else 'min-scaled'

train_DS_t5 = reader.HigDim_AnnDS(AnnData=adata,
                              n_dimension=n_dimension,
                              n_timepoint=5,  
                              log_transform=log_transform,
                              norm_time = False,
                              cellstate_key="DM_EigenVectors",  #'DM_EigenVector'
                              resampling_rate = resampling_rate,
                              )

u_b_t5 = train_DS_t5.u_b.cpu().numpy().reshape(5, -1)



train_DS_t7 = reader.HigDim_AnnDS(AnnData=adata,  n_timepoint=7,   
                                  n_dimension=n_dimension,
                                  log_transform=log_transform,
                                  cellstate_key="DM_EigenVectors",  #'DM_EigenVector'
                                  resampling_rate = resampling_rate,
                                  )

u_b_t7 = train_DS_t7.u_b.cpu().numpy().reshape(7, -1)



train_DS_t9 = reader.HigDim_AnnDS(AnnData=adata,   
                                  n_dimension=n_dimension,
                                  log_transform=log_transform,
                                  cellstate_key="DM_EigenVectors",  #'DM_EigenVector'
                                  resampling_rate = resampling_rate,
                                  )

u_b_t9 = train_DS_t9.u_b.cpu().numpy().reshape(9, -1)

# PINN.pl.params_in_umap(train_DS_t5.adata, u_b_t5, param='u', copy=False, cell_of_t=False)
# PINN.pl.params_in_umap(train_DS_t7.adata, u_b_t7, param='u',cell_of_t=False)
# PINN.pl.params_in_umap(train_DS_t9.adata, u_b_t9, param='u')


print("  ".join(["{:.2}".format(v) for v in train_DS_t9.popD['mean']]))
plt.figure(dpi=300)
plt.bar(range(9), train_DS_t9.popD['mean'], width=0.8)
plt.xticks(range(9), timepoints)
plt.ylabel(f'{processing} population size', fontsize=12)
plt.xlabel("timepoints (days)", fontsize=12)


fig,axs = plt.subplots(2,1,figsize=(5,5), dpi=300, sharex=True)

axs[0].plot(np.log10(u_b_t9.max(axis=1)), marker='o', label='t9')
axs[0].plot(np.log10(u_b_t5.max(axis=1)), marker='o', label='t5')
axs[0].plot(np.log10(u_b_t7.max(axis=1)), marker='o', label='t7')
axs[0].set_ylabel('max density', fontsize=12)
y_ticks = axs[0].get_yticks()
axs[0].set_yticklabels(
    [np.format_float_scientific(v,precision=1) for v in np.power(10, y_ticks)],
    fontsize=10)
axs[0].legend()

axs[1].plot(np.log10(u_b_t9.min(axis=1)), marker='o', label='t9')
axs[1].plot(np.log10(u_b_t5.min(axis=1)), marker='o', label='t5')
axs[1].plot(np.log10(u_b_t7.min(axis=1)), marker='o', label='t7')
axs[1].set_ylabel('min density', fontsize=12)

axs[1].set_xticks(range(9))
axs[1].set_xticklabels(timepoints)
axs[1].set_xlabel("timepoints (days)", fontsize=12)
y_ticks = axs[1].get_yticks()
axs[1].set_yticklabels(
    [np.format_float_scientific(v,precision=1) for v in np.power(10, y_ticks)],
    fontsize=10)
axs[1].legend()

fig.suptitle(
    f"the density with {n_dimension} dimensional DM \n & {processing} population size",
    fontsize=13)


## resampling



# # expand the u_b_t5 
# density_funs = train_DS_t9.density_funs

# ext_u_from_t5 = [u_b_t5]  # u of t5

# for i_t in [5, 6,7,8]:
#     print(f'i_t {i_t} is', train_DS_t9.popD['t'][i_t])

#     ub_ext = density_funs[i_t](train_DS_t5.cellstate.T)
#     ub_ext = ub_ext / ub_ext.sum()
#     ub_ext = ub_ext * train_DS_t9.popD['mean'][i_t]

#     ext_u_from_t5.append(ub_ext)

# u_b_extend = np.vstack(ext_u_from_t5)
# u_different = np.diff(u_b_extend, axis=0)

# PINN.pl.params_in_umap(train_DS_t5.adata, u_different[:5], param='u_rolling_diff', cell_of_t=False);
# PINN.pl.params_in_umap(train_DS_t5.adata, u_different[:5], param='u_rolling_diff', cell_of_t=True);