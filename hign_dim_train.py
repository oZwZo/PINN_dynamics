import os, argparse, typing
import numpy as np
import pandas as pd
import scanpy as sc
import torch 
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from functools import partial
import pytorch_lightning as pl
from pytorch_lightning import callbacks 
from pytorch_lightning import loggers as pl_loggers

from PINN import models as models
from PINN import reader 
from PINN import functions as fns

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


parser = argparse.ArgumentParser("Training PINN dynamics on example dataset")
parser.add_argument("-D", "--dataset", type=str, required=False, default="HSPC_clu7", help='the name of the dataset, can be found under folder data')
parser.add_argument("-K", "--cellstate_key", type=str, required=False, default="cellstate", help='the obsm key on which we represent cell and compute density')
parser.add_argument("-M", "--model", type=str, required=False, default="Cspline_PINN", help='the model class, defined in models.py')
parser.add_argument("-W", "--pretrained", type=str, required=False, default=None, help='the path of the pretrained weights')
parser.add_argument("-G", "--gpu_devices", type=int, required=True, default=None, help='select which gpu devices to use')
parser.add_argument("--lr", type=float, required=False, default=3e-3, help='the learning rate for training the model')
parser.add_argument("--schedule_lr", type=str, required=False, default="StepLR", help='LambdaLR if passing a lambda expression, else StepLR')
parser.add_argument("--n_dimension", type=int, required=False, default=5, help='the number of dimension to used for estimating density')
parser.add_argument("--nearby_cellstate", type=int, required=False, default=10, help='the number of nearby cell state to include within a minibatch')
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

save_path = os.path.join(main_path, 'logs', f"{args.dataset}-{args.cellstate_key}_multiBnch", args.model)
if not os.path.exists(save_path):
    os.mkdir(save_path)


# train_DS = reader.MeshGrid_DS(Data_pt=pt_path,  n_grid=args.n_grid,  nearby_cellstate=args.nearby_cellstate, collocation_points=300, n_repeat=10)

ery_mk_ad = sc.read_h5ad(h5_path)

# MeshGrid_Resample
# MeshGrid_logDS
train_DS = reader.HigDimRe_AnnDS(AnnData=ery_mk_ad, n_dimension = 5, cellstate_key=args.cellstate_key,  #'Actb_Kcnn4_scaled_S'
                                     nearby_cellstate=args.nearby_cellstate, 
                                    collocation_points=300)

batch_size = 50 if args.nearby_cellstate == 1 else 1
train_DL = DataLoader(train_DS, batch_size=batch_size, num_workers=10, shuffle=True)


                            ###                  ###
                            #     lr scheduler     #
                            ###                  ###

# pseudo dynamics model
if args.schedule_lr == 'StepLR':
    schedule_lr = partial(lr_scheduler.StepLR, step_size  = 100 , gamma = 0.5)

elif args.schedule_lr == 'CyclicLR':
    schedule_lr = partial(lr_scheduler.CyclicLR, base_lr=args.lr, max_lr=10*args.lr)

elif args.schedule_lr == 'CosineAnnealingLR':
    schedule_lr = partial(lr_scheduler.CosineAnnealingLR, T_max = 100)

elif args.schedule_lr == 'CosineAnnealingWarmRestarts':
    schedule_lr = partial(lr_scheduler.CosineAnnealingWarmRestarts, T_0 = 3)

elif args.schedule_lr in dir(torch.optim.lr_scheduler):
    # suitable for some sch like `LinearLR` `PolynomialLR`
    schedule_lr = eval("lr_sceduler%s" %args.schedule_lr)

elif isinstance(eval(args.schedule_lr), typing.Callable):
    # customize schedu_lr
    rule = eval(args.schedule_lr)
    schedule_lr = partial(lr_scheduler.LambdaLR, lr_lambda = rule)

elif args.schedule_lr in dir(fns):
    # pre-defined strategy , can be `Lambda1` or `Lambda2``
    rule = eval(f"fns.{args.schedule_lr}")
    schedule_lr = partial(lr_scheduler.LambdaLR, lr_lambda = rule)

else:
    # pass the String
    schedule_lr = args.schedule_lr


                            ###                  ###
                            #     define model     #
                            ###                  ###


# define neural network surrogate

# if args.pretrained is not None:
#     model = models.MLP.load_from_checkpoint(args.pretrained)
# else:
#     model = models.MLP(
#         lr=3e-4,
#         channels = channels,
#         activation_fn='Tanh'
#     )

channels = [int(c) for c in args.channels.split(",")]   
u_theta = models.MLP_surrogate(channels = channels, activation_fn='Tanh')
Model_Class = eval(f"models.{args.model}")
model = Model_Class(u=u_theta, n_dim=5, 
                         n_knot=9, lr=args.lr, 
                         schedule_lr=schedule_lr)


                            ###                     ###
                            #     define Triainer     #
                            ###                     ###
                            
# device
device = 'gpu' if torch.cuda.is_available() else 'cpu'
device = 'cpu' if args.gpu_devices == None else 'gpu'
gpu_device = args.gpu_devices

# logger and checkpoints

tb_logger = pl_loggers.TensorBoardLogger(save_dir=save_path)

# trainer
trainer = pl.Trainer(
                    #auto_lr_find=True,
                    accelerator=device,
                    # fast_dev_run=True,
                    gradient_clip_val=0.5,
                    default_root_dir=save_path,
                    logger=tb_logger,
                    devices = [gpu_device],
                    max_epochs=300,
                    callbacks=[callbacks.ModelCheckpoint(filename='{epoch}-{total_loss:.8f}',
                                                monitor="total_loss", mode="min", save_top_k=2)]
                    )

# start training
model.train()
trainer.fit(model, train_DL, ckpt_path = args.pretrained)
