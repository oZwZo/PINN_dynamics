import os, argparse
import numpy as np
import pandas as pd
import torch 
from torch import nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import callbacks 
from pytorch_lightning import loggers as pl_loggers

import models as models
import reader 

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


parser = argparse.ArgumentParser("Training PINN dynamics on example dataset")
parser.add_argument("-M", "--model", type=str, required=False, default="Cspline_PINN", help='the model class, defined in models.py')
parser.add_argument("-W", "--pretrained", type=str, required=False, default=None, help='the path of the pretrained weights')
parser.add_argument("-G", "--gpu_devices", type=int, required=True, default=None, help='select which gpu devices to use')
args = parser.parse_args()


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


train_DS = reader.Random_ExtractDataset(Data_pt=pt_path, n_time=10, n_grid=300, collocation_points=300, n_repeat=10)

# val_DS= reader.Pdyn_ExtractDataset(Data_pt=pt_path, n_grid=300, collocation_points=300, n_repeat=1)

train_DL = DataLoader(train_DS, batch_size=1, num_workers=10, shuffle=True)
# val_DL = DataLoader(val_DS, batch_size=1, num_workers=4)



                            ###                  ###
                            #     define model     #
                            ###                  ###

# define neural network surrogate
u_theta = models.MLP_surrogate(channels = [2, 32, 8, 1], activation_fn='Tanh')

# pseudo dynamics model
Model_Class = eval(f"models.{args.model}")
Pdyn_model = Model_Class(u=u_theta, n_knot=9, lr=3e-4)

if args.pretrained is not None:
    assert os.path.exists(args.pretrained), "pretrained weights not found"
    Pdyn_model = models.Cspline_PINN.load_from_checkpoint(args.pretrained)


                            ###                     ###
                            #     define Triainer     #
                            ###                     ###
                            
# device
device = 'gpu' if torch.cuda.is_available() else 'cpu'
device = 'cpu' if args.gpu_devices == None else 'gpu'
gpu_device = args.gpu_devices

# logger and checkpoints
pth_save_path = os.path.join(main_path, f"logs/{args.model}_RandMiniB/")
tb_logger = pl_loggers.TensorBoardLogger(save_dir=pth_save_path)

# trainer
trainer = pl.Trainer(
                    #auto_lr_find=True,
                    accelerator=device,
                    # fast_dev_run=True,
                    default_root_dir=pth_save_path,
                    logger=tb_logger,
                    devices = [gpu_device],
                    max_epochs=300,
                    callbacks=[callbacks.ModelCheckpoint(filename='{epoch}-{total_loss:.8f}',
                                                monitor="total_loss", mode="min", save_top_k=2)]
                    )

# start training
Pdyn_model.train()
trainer.fit(Pdyn_model, train_DL)
