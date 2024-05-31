import os, argparse
import numpy as np
import pandas as pd
import torch 
from torch import nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import callbacks 
from pytorch_lightning import loggers as pl_loggers

from PINN import models as models
from PINN import reader 

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


parser = argparse.ArgumentParser("Training PINN dynamics on example dataset")
parser.add_argument("-D", "--dataset", type=str, required=False, default="HSPC_clu7", help='the name of the dataset, can be found under folder data')
parser.add_argument("-M", "--model", type=str, required=False, default="Cspline_PINN", help='the model class, defined in models.py')
parser.add_argument("-W", "--pretrained", type=str, required=False, default=None, help='the path of the pretrained weights')
parser.add_argument("-G", "--gpu_devices", type=int, required=True, default=None, help='select which gpu devices to use')
parser.add_argument("--lr", type=float, required=False, default=3e-3, help='the learning rate for training the model')
parser.add_argument("--n_grid", type=int, required=False, default=300, help='the number of grid or h to devid the cell state space')
parser.add_argument("--channels", type=str, required=False, default="2,32,32,1", help='the depth and width of the model')
args = parser.parse_args()


                        ###               ###
                        #     read data     #
                        ###               ###

path = os.path.abspath(".")
pt_path = os.path.join(path, f'{args.dataset}.pt')

if not os.path.exists(pt_path):
    main_path = path
    pt_path = os.path.join(path, f'data/{args.dataset}.pt')
else:
    main_path = os.path.dirname(path)

save_path = os.path.join(main_path, 'logs', args.dataset)
if not os.path.exists(save_path):
    os.mkdir(save_path)


train_DS = reader.Random_ExtractDataset(Data_pt=pt_path, n_time=10, n_grid=args.n_grid, collocation_points=300, n_repeat=10)


train_DL = DataLoader(train_DS, batch_size=1, num_workers=20, shuffle=True)



                            ###                  ###
                            #     define model     #
                            ###                  ###

# define neural network surrogate
channels = [int(c) for c in args.channels.split(",")]
u_theta = models.MLP_surrogate(channels = channels, activation_fn='Tanh')

# pseudo dynamics model
Model_Class = eval(f"models.{args.model}")
Pdyn_model = Model_Class(u=u_theta, n_grid=args.n_grid, n_knot=9, lr=args.lr)

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
pth_save_path = os.path.join(main_path, "logs", f"{args.dataset}/{args.model}_RandMiniB/")
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
