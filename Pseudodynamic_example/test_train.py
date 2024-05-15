import os
import numpy as np
import pandas as pd
import torch 
from torch import nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import callbacks 
from pytorch_lightning import loggers as pl_loggers

import models as models
from reader import Pdyn_ExtractDataset





###               ###
#     read data     #
###               ###

path = os.path.abspath(".")
pt_path = os.path.join(path, 'dataExample.pt')

if not os.path.exists(pt_path):
    main_path = path
    pt_path = os.path.join(path, 'Pseudodynamic_example/dataExample.pt')
else:
    main_path = os.path.dirname(path)

train_DS = Pdyn_ExtractDataset(Data_pt=pt_path, n_grid=300, collocation_points=300, n_repeat=10)
val_DS= Pdyn_ExtractDataset(Data_pt=pt_path, n_grid=300, collocation_points=600, n_repeat=1)

train_DL = DataLoader(train_DS, batch_size=1, num_workers=4)
val_DL = DataLoader(val_DS, batch_size=1, num_workers=4)



###                  ###
#     define model     #
###                  ###

# define neural network surrogate
u_theta = models.MLP_surrogate(channels = [2, 32, 32, 32, 1], activation_fn='Tanh')

# pseudo dynamics model
Pdyn_model = models.Cspline_PINN(u=u_theta, n_knot=11, lr=3e-3)

# pretrained
# ckpt_path = os.path.join(main_path, "logs/Cspline_PINN/lightning_logs/version_3/checkpoints/epoch=297-total_loss=10.55338860.ckpt")
# Pdyn_model = models.Cspline_PINN.load_from_checkpoint(ckpt_path)


###                     ###
#     define Triainer     #
###                     ###

device = 'gpu' if torch.cuda.is_available() else 'cpu'
gpu_device = 2
pth_save_path = os.path.join(main_path, "logs/Cspline_PINN/")
tb_logger = pl_loggers.TensorBoardLogger(save_dir=pth_save_path)

trainer = pl.Trainer(auto_lr_find=True,
                     accelerator=device,
                     # fast_dev_run=True,
                     default_root_dir=pth_save_path,
                     logger=tb_logger,
                     devices = [gpu_device],
                     max_epochs=300,
                     callbacks=[callbacks.ModelCheckpoint(filename='{epoch}-{total_loss:.8f}',
                                                  monitor="total_loss", mode="min", save_top_k=2)]
                     )

# callbacks=[callbacks.ModelCheckpoint(filename='{epoch}-{val_cre:.8f}',
#                                                   monitor="val_cre", mode="min", save_top_k=2),
#                                 callbacks.EarlyStopping(monitor="val_cre", mode="min", patience=20),]



Pdyn_model.train()
trainer.fit(Pdyn_model, train_DL)
