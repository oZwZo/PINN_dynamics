import os, argparse, typing

import numpy as np
import pandas as pd
import scanpy as sc
import torch 
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from functools import partial
import pytorch_lightning as pl
from pytorch_lightning import callbacks 
from pytorch_lightning import loggers as pl_loggers

from PINN import models as models
from PINN import reader 
from PINN import functions as fns

torch.set_float32_matmul_precision('medium')

    

parser = argparse.ArgumentParser("Training PINN dynamics on single timepoint dataset")
parser.add_argument("-D", "--dataset", type=str, required=False, default="HSPC_clu7", help='the name of the dataset, can be found under folder data')
parser.add_argument("-K", "--cellstate_key", type=str, required=False, default="cellstate", help='the obsm key on which we represent cell and compute density')
parser.add_argument("-M", "--model", type=str, required=False, default="Cspline_PINN", help='the model class, defined in models.py')
parser.add_argument("-W", "--pretrained", type=str, required=False, default=None, help='the path of the pretrained weights')
parser.add_argument("-G", "--gpu_devices", type=int, required=True, default=None, help='select which gpu devices to use')
parser.add_argument("--lr", type=float, required=False, default=3e-3, help='the learning rate for training the model')
parser.add_argument("--schedule_lr", type=str, required=False, default="StepLR", help='LambdaLR if passing a lambda expression, else StepLR')
parser.add_argument("--n_grid", type=int, required=False, default=300, help='the number of grid or h to devid the cell state space')
parser.add_argument("--batch_size", type=int, required=False, default=10, help='the number of nearby cell state to include within a minibatch')
parser.add_argument("--channels", type=str, required=False, default="3,32,32,1", help='the depth and width of the model')
args = parser.parse_args()


                                ###               ###
                                #     read data     #
                                ###               ###

path = os.path.abspath(".")
h5_path = os.path.join(path, f'{args.dataset}.h5ad')

if not os.path.exists(h5_path):
    main_path = path
    h5_path = os.path.join(path, f'data/{args.dataset}.h5ad')
else:
    main_path = os.path.dirname(path)

save_path = os.path.join(main_path, 'logs', "Simple_DS_MLP", "MLP")
if not os.path.exists(save_path):
    os.mkdir(save_path)


ery_mk_ad = sc.read_h5ad(h5_path)
channels = [int(c) for c in args.channels.split(",")]   

model = models.MLP(
    lr=3e-4,
    channels = channels,
    activation_fn='Tanh'
)

train_DS =  Simple_DS(n_timepoint = 3, AnnData=ery_mk_ad, cellstate_key='Actb_Kcnn4_scaled_S',
                                    n_grid=args.n_grid,   # nearby cell state is not used 
                                    collocation_points=300, n_repeat=2)

train_DL = DataLoader(train_DS, batch_size=args.batch_size, shuffle=True, num_workers=20)

device = 'gpu' if torch.cuda.is_available() else 'cpu'
device = 'cpu' if args.gpu_devices == None else 'gpu'
gpu_device = args.gpu_devices

logs_dir = "/ssd/users/Wergillius/Project/PINN_dynamics/logs"
save_path = f"{logs_dir}/Simple_DS_MLP"

trainer = pl.Trainer(
                    #auto_lr_find=True,
                    accelerator=device,
                    # fast_dev_run=True,
                    gradient_clip_val=0.5,
                    default_root_dir=save_path,
                    devices = [gpu_device],
                    max_epochs=300,
                    callbacks=[callbacks.ModelCheckpoint(filename='{epoch}-{total_loss:.8f}',
                                                monitor="total_loss", mode="min", save_top_k=2)]
                    )

trainer.fit(model, train_dataloaders=train_DL)