import os
import numpy as np
import pandas as pd
import torch 
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from reader import Pdyn_ExtractDataset
import PINNs




###               ###
#     read data     #
###               ###

pt_path = '/home/wergillius/Project/PINN_dynamics/Pseudodynamic_example/dataExample.pt'
train_DS = Pdyn_ExtractDataset(Data_pt=pt_path, n_grid=300, collocation_points=300, n_repeat=10)
val_DS= Pdyn_ExtractDataset(Data_pt=pt_path, n_grid=300, collocation_points=600, n_repeat=1)

train_DL = DataLoader(train_DS, batch_size=1, num_workers=4)
val_DL = DataLoader(val_DS, batch_size=1, num_workers=4)



###                  ###
#     define model     #
###                  ###

# define neural network surrogate
u_theta = PINNs.MLP_surrogate(channels = [2, 32, 32, 1], activation_fn='Tanh')

# pseudo dynamics model
Pdyn_model = PINNs.Cspline_PINN(u=u_theta, n_knot=11)




###                     ###
#     define Triainer     #
###                     ###
device = 'gpu' if torch.cuda.is_available() else 'cpu'
gpu_device = 0
pth_save_path = "../logs/Cspline_PINN"


trainer = pl.Trainer(auto_lr_find=True,
                     accelerator=device,
                     # fast_dev_run=True,
                     default_root_dir=pth_save_path,
                     devices = [gpu_device],
                     max_epochs=100,
                     )

# callbacks=[callbacks.ModelCheckpoint(filename='{epoch}-{val_cre:.8f}',
#                                                   monitor="val_cre", mode="min", save_top_k=2),
#                                 callbacks.EarlyStopping(monitor="val_cre", mode="min", patience=20),]
    

Pdyn_model.train()
trainer.fit(Pdyn_model, train_DL, val_dataloaders=val_DL)
        