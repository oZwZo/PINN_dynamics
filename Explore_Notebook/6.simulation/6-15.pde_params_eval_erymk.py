
# %%
# %load_ext autoreload
# %autoreload 2
import os, sys, re
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import torch
import time
import PINN
from PINN import reader, models, pl, tl
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torchdiffeq import odeint

import matplotlib as mpl
import seaborn as sns
from matplotlib.patches import Patch


os.chdir("/ssd/users/Wergillius/Project/PINN_dynamics")
sc.settings.set_figure_params(frameon=False, dpi=70, figsize=(3,3))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("script for evaluting ery_mk data fitting")
    parser.add_argument("--config_dir", type=str, required=True, default=None, help='folder of existing config JSON files')
    args = parser.parse_args()
    config_dir = args.config_dir
else:
    config_dir = ""

# %%
# find all config files

configs = [file for file in os.listdir(config_dir) if file.endswith('.json')]

# %%
# load config for different runs
config_dict = {}
model_dict = {}

for js in configs:
    version = js.split("_")[0][1:] # after V
    config_dict[version] = PINN.ExperimentConfig(os.path.join(config_dir, js))

for v,config in config_dict.items():
    print(v)
    ckpt = config.find_lastest_ckpt()
    print(ckpt)
    print("\n")
    model_dict[v] = PINN.models.pde_params.load_from_checkpoint(ckpt, map_location='cpu')

    result_dir = config.experiment_config['save_dir'].replace("logs", "results")
    result_dir = result_dir + f"_{v}"
    config.result_dir = result_dir
    PINN.tl.make_dir(result_dir)

# %%
# # load dataset
dataset_name = config.experiment_config['dataset']
print(f'\nLoading dataset: {dataset_name}')
adata = sc.read_h5ad(f'data/{dataset_name}.h5ad')

# %%
ds_config = config.dataset_config.copy()
ds_config['timepoint_idx'] = None
ds_config['knn_volume'] = eval(config.raw_args['knn_volume'])

full_DS = PINN.reader.TwoTimpepoint_AnnDS(adata,split=None,**ds_config)
cellstate_key = config.dataset_config['cellstate_key']


# %%
v_dict = {}
for v in model_dict.keys():

    pde_model = model_dict[v]
    config = config_dict[v]

    g_pred_ay, v_pred_ay, D_pred_ay = pde_model.predict_param(full_DS)

    result_dir = config_dict[v].result_dir
    

    print(f"model version {v}, result saved to :")
    print(result_dir)

    fig_g,axs = PINN.pl.params_in_umap(adata, g_pred_ay, param=r'$g$')
    fig_g.savefig(os.path.join(result_dir,'g.png'), dpi=150, transparent=True)
    

    if D_pred_ay.shape[-1] == 1:
        fig_D,axs = PINN.pl.params_in_umap(adata, D_pred_ay.squeeze(), param=r'$D$')
        fig_D.savefig(os.path.join(result_dir,'D.png'), dpi=150, transparent=True)
        

    v_pred_norm = np.sqrt(np.sum(v_pred_ay**2, axis=-1))
    fig_v,axs = PINN.pl.params_in_umap(adata, v_pred_norm, param=r'$v$')
    fig_v.savefig(os.path.join(result_dir,'v.png'), dpi=150, transparent=True)
    

    # visualize like RNA velocity
    cellstate_ad = PINN.tl.make_coord_adata(adata, cellstate_key=cellstate_key, n_dimension=config.dataset_config['n_dimension'], v = v_pred_ay)
    vkeys = [k for k in list(cellstate_ad.layers.keys()) if k.endswith("v")]

    for vkey in vkeys:
        # compute velocity graph
        scv.tl.velocity_graph(cellstate_ad,  vkey=vkey, xkey='cellstate', n_jobs=20)
        # vis
        fig_velocity = plt.figure(dpi=120, figsize=(4,4))
        ax = fig_velocity.gca()
        scv.pl.velocity_embedding_stream(cellstate_ad, color='anno_man', vkey=vkey, 
                                        basis='umap', ax=ax, 
                                        legend_loc='right', alpha=0.01,
                                        title=vkey, 
                                        save=f"{result_dir}/{vkey}_velo.png")


# ploting simulation

performance_js = []
timepoints = adata.uns['pop']['t']
n_timepoints = timepoints.shape[0]

for v in model_dict:
    device = 'cuda:4'
    pde_model = model_dict[v].to(device).eval()
    u_b = full_DS.u_b.cpu().numpy()

    u_sim = PINN.tl.density_shortterm_simulation(pde_model, DataSet=full_DS, timepoints=adata.uns['pop']['t'])
    u_int_all = np.concatenate([u_b[0,None], u_sim], axis=0)

    fig_ub,axs = PINN.pl.params_in_umap(adata, u_b, param=r'$u_b$')
    fig_ub.savefig(os.path.join(result_dir,'u_b.png'), dpi=150, transparent=True)

    fig_uint,axs = PINN.pl.params_in_umap(adata, u_b, param=r'$u_{sim}$')
    fig_uint.savefig(os.path.join(result_dir,'u_int.png'), dpi=150, transparent=True)

    fig_loguint,axs = PINN.pl.params_in_umap(adata, np.log(u_b+1e-10), param=r'$\log {u_{sim}}$')
    fig_loguint.savefig(os.path.join(result_dir,'log_u_sim.png'), dpi=150, transparent=True)

    KLD_ls = PINN.tl.KLD_density(u_b, u_int_all)
    W1 = PINN.tl.W_distance(u_b, u_int_all, p=1)
    W2 = PINN.tl.W_distance(u_b, u_int_all)

    df = pd.DataFrame({
        "v":[v]*n_timepoints, 
        'KLD':KLD_ls,
        'W1': W1, 
        'W2':W2,
        't':timepoints
        })
    performance_js.append(df)

# concat all df
density_performance_df = pd.concat(performance_js, axis=0)
density_performance_df.to_csv(os.path.join(os.path.dirname(result_dir),'density_performance.csv'))