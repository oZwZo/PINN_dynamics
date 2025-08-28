
# %%
# %load_ext autoreload
# %autoreload 2
import os, sys, re
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import torch
import torch.nn.functional as F
from PINN.models import Density_Transfer
from tqdm.auto import tqdm

import time
import PINN
from PINN import reader, models, pl, tl
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torchdiffeq import odeint

import matplotlib as mpl
import seaborn as sns
from matplotlib.patches import Patch


# os.chdir("/home/wz369/rds/hpc-work/PINN_dynamics")
os.chdir("/Users/weizhongzheng/Documents/python_project/PINN_dynamics")
sc.settings.set_figure_params(frameon=False, dpi=70, figsize=(3,3))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("script for evaluting ery_mk data fitting")
    parser.add_argument("--config_dir", type=str, required=True, default=None, help='folder of existing config JSON files')
    parser.add_argument("--celltype_key", type=str, required=False, default='anno_man', help='the adata obs key defining cell type')
    parser.add_argument("--start_celltype", type=str, required=False, default='HSC', help='the cell type subset to use as starting point to simualte trajectory')
    parser.add_argument("--transport_time", type=int, required=False, default=None, help='the period of use to do simulation, if none , the next timeepoint will be used')
    parser.add_argument("--n_interval", type=int, required=False, default=10, help='the nubmer of time interval along the simulated trajectory')
    parser.add_argument("--n_cell", type=int, required=False, default=300, help='the minibatch size')
    parser.add_argument("-G", "--GPU", type=str, required=False, default="0", help='the GPU device to use')
    args = parser.parse_args()

# %%
# find all config files
config_dir = os.path.abspath(args.config_dir)
configs = [file for file in os.listdir(config_dir) if file.endswith('.json')]
device = torch.device(f"cuda:{args.GPU}" if torch.cuda.is_available() else "cpu")
# %%
# load config for different runs
config_dict = {}
model_dict = {}

for js in configs:
    version = js.split("_")[0][1:] # after V
    config = PINN.ExperimentConfig(os.path.abspath(os.path.join(config_dir, js)))
    main_dir = "/Users/weizhongzheng/Documents/python_project/PINN_dynamics/"
    config.from_json(os.path.abspath(os.path.join(config_dir, js)), main_dir)
    config_dict[version] = config

for v,config in config_dict.items():
    print('version', v)
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
print(f'loading dataset: {dataset_name}')
adata = sc.read_h5ad(f'data/{dataset_name}.h5ad')
HSC_ad = adata[adata.obs['anno_man']=='HSC'].copy()

# %%
ds_config = config.dataset_config.copy()
timepoint_idx = ds_config['timepoint_idx'].copy()
ds_config['timepoint_idx'] = None
ds_config['knn_volume'] = eval(config.raw_args['knn_volume'])

full_DS = PINN.reader.TwoTimpepoint_AnnDS(adata,split=None,**ds_config)
cellstate_key = config.dataset_config['cellstate_key']


# %%

n_interval = args.n_interval
n_cell = args.n_cell
celltype_key = args.celltype_key
start_celltype = args.start_celltype


for v in model_dict.keys():

    print(f'conducting density transport for version {v}')

    config = config_dict[v]
    result_dir = config.result_dir
    PINN.tl.make_dir(os.path.join(result_dir, "Density_transport"))

    # initiate Density transfer class
    pde_model = model_dict[v].to(device)
    DT = Density_Transfer(pde_model)

    # timepoint key
    timepoint_key = 'timepoint_tx_days'
    timepoint_tx_days = np.array(sorted(adata.obs[timepoint_key].unique()))
    t0 = timepoint_tx_days[0]
    scaling_factor = pde_model.time_scale_factor
    HSC_cbs = []

    selected_time = np.array(timepoint_tx_days)[timepoint_idx][:-1]
    for it,t in tqdm(enumerate(selected_time)):

        if  args.transport_time is not None:
            t2 = t + args.transport_time
        else:
            t2 = timepoint_tx_days[it+1]
        
        if ds_config['norm_time']=='min_minus':
            t1 = t - t0         # starting timepoint for simulation
            t2 = t2 - t0        # ending timepoint for simulation
        else:
            t1 = t/t0
            t2 /= t0
            
        integrate_time = np.linspace(t1, t2 ,n_interval+1) / pde_model.time_scale_factor
        
        
        try:
            # print out the integration time
            print(np.round(integrate_time, 2))
        except:
            pass

        # find cell of time
        start_cell = adata.obs.query(f"`{celltype_key}` == @start_celltype & `{timepoint_key}` == @t").index
        start_cell = list(start_cell)
        HSC_cbs.append(start_cell)

        # define initital density and cellstates
        cell_index = [np.where(adata.obs_names == x)[0].item() for x in start_cell]
        u0 = full_DS.u_b[it, cell_index].float().to(device)
        s0 = torch.from_numpy(full_DS.cellstate[cell_index]).float().to(device)

        S_trajectory = DT.cellstate_drift(s0, integrate_time)
        Tmaps_t, Tmaps_t_norm = DT.transition_by_batch(s0, u0, integrate_time, n_interval=n_interval, ncell=n_cell)

        del u0, s0


        print(f"results saved to {result_dir}")
        endtime = np.round(integrate_time, 2)[-1]
        np.save(os.path.join(result_dir, "Density_transport", f"{start_celltype}_Day{t}-{endtime}_sim_trajectory.npy"), S_trajectory)
        np.save(os.path.join(result_dir, "Density_transport", f"{start_celltype}_Day{t}-{endtime}_TransportMap.npy"), Tmaps_t)
        np.save(os.path.join(result_dir, "Density_transport", f"{start_celltype}_Day{t}-{endtime}_Norm_TransportMap.npy"), Tmaps_t_norm)
        np.save(os.path.join(result_dir, "Density_transport", f"{start_celltype}_Day{t}-{endtime}_cellbarcode.npy"), start_cell)
    
    
    # adata_t_select = [adata.obs.query("`timeponit_tx_days` in @selected_time").index]
    DT = PINN.models.DT_analysis(adata, result_dir=result_dir)
    celltype_trajectory = DT.annotate_trajectory('DM_EigenVectors_multiscaled', obs_key='anno_man', copy=False)

    for t, df in celltype_trajectory.items():
        df.to_csv(os.path.join(result_dir, "Density_transport", f"{start_celltype}_Day{t}_ct_prop.csv"),index=True)
    
    
    print(f"cell type propotion for version {v} is saved")
print("Done")
    
