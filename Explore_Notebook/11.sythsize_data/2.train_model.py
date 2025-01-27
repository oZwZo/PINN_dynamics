import os,sys
import yaml
import torch
import numpy as np
import pandas as pd 

from PINN import models, reader
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

os.chdir("/ssd/users/Wergillius/Project/PINN_dynamics")


log_dir = "logs/5Dim_ncs_syn_Jan23_0-4_time_longer"


def check_existing_version(log_dir = log_dir):
    # check lightning model version
    try:
        existing_v = os.listdir(f"{log_dir}/lightning_logs")
        v_num = [int(v.split("_")[-1]) for v in existing_v]
        if len(v_num)==0:
            max_v = -1
        else:
            max_v = max(v_num)
    except FileNotFoundError:
        max_v = -1

    # check config version
    config_files = [f for f in os.listdir(log_dir) if f.endswith("_config.yaml")]
    v_config = [int(v.split("_")[0][1:]) for v in config_files]
    if len(v_config)==0:
        max_v_config = -1
    else:
        max_v_config = max(v_config)

    assert max_v == max_v_config, "two version must be the same"
    return max_v_config

    


syn_init_ckpt = torch.load("data/synthesized_data/5Dim_ncs_synparam_Jan23/syn_init_condition_0-4.ckpt")
cellstate = syn_init_ckpt['cellstate']

syn_result_ckpt = torch.load("data/synthesized_data/5Dim_ncs_synparam_Jan23/syn_result_0-4.ckpt")
density = syn_result_ckpt['u_simulate']
integrate_time = syn_result_ckpt['integrate_time']
ds = syn_result_ckpt['delta_x']


dataset_kws = dict(
    seen_timepoints = [0, 1, 2, 4, 6, 8,10],
    leaveout_timepoints = [5,7],
    test_timepoitns = [3,9],
    batchsize = 128
)

train_DS = reader.Syn_DS(cellstate=cellstate, 
                  density=density[dataset_kws['seen_timepoints']],
                  integrate_time= integrate_time[dataset_kws['seen_timepoints']],
                  deltax=ds,
                  batchsize = dataset_kws['batchsize']
                  )

val_DS = reader.Syn_DS(cellstate=cellstate, 
                  density=density[dataset_kws['leaveout_timepoints']],
                  integrate_time= integrate_time[dataset_kws['leaveout_timepoints']],
                  deltax=ds,
                  batchsize = 512)

train_DL = DataLoader(train_DS, batch_size=None, num_workers=5)
val_DL = DataLoader(val_DS, batch_size=None, num_workers=5)


model_kws = dict(
    channels =   [5+1, 16, 16, 1],
    g_channels = [6, 16, 16, 1],
    v_channels = [6, 32, 16, 5],
    D_channels = [6, 32, 1],
    activation_fn='Tanh',
    ode_tol = 1e-4,
    D_penalty = 0.1,
    deltax_weight = 1,
    weight_intensity = 3,
    time_scale_factor = 1,
    time_sensitive = True,
    growth_weight = 3,
)
model = models.pde_params(**model_kws)


config = model_kws
config.update(dataset_kws)


version = check_existing_version(log_dir) + 1

with open(f"{log_dir}/v{version}_config.yaml", 'w') as outfile:
    yaml.dump(config, outfile, default_flow_style=False)


print("cofig saved to ", f"v{version}_config.yaml")

trainer = pl.Trainer(
                    enable_progress_bar=False,
                    auto_lr_find=True,
                    accelerator='gpu',
                    default_root_dir=log_dir,
                    devices = [3], 
                    max_epochs=300,
                    callbacks=[pl.callbacks.ModelCheckpoint(
                                filename='{epoch}-{val_loss:.8f}',
                                monitor="total_loss", mode="min", save_top_k=2)]
                    )

trainer.fit(model, train_dataloaders=train_DL, val_dataloaders=val_DL)