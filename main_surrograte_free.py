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

import PINN
from PINN import models as models
from PINN import reader 
from PINN import functions as fns


torch.set_float32_matmul_precision('medium')

    

parser = argparse.ArgumentParser("Training PINN dynamics on mesh-free high dimensional cellstate, with surrogate-free training scheme")
parser.add_argument("-D", "--dataset", type=str, required=False, default="HSPC_clu7", help='the name of the dataset, can be found under folder data')
parser.add_argument("-K", "--cellstate_key", type=str, required=False, default="cellstate", help='the obsm key on which we represent cell and compute density')
parser.add_argument("-M", "--model", type=str, required=False, default="pde_u_free", help='the model class, defined in models.py')
parser.add_argument("-W", "--pretrained", type=str, required=False, default=None, help='the path of the pretrained weights')
parser.add_argument("-G", "--gpu_devices", type=int, required=True, default=None, help='select which gpu devices to use')
parser.add_argument("-L",  "--log_name", type=str, required=False, default=None, help='the name of the logging directory')
parser.add_argument("--lr", type=float, required=False, default=3e-4, help='the learning rate for training the model')
parser.add_argument("--schedule_lr", type=str, required=False, default="CyclicLR", help='LambdaLR if passing a lambda expression, else StepLR')
parser.add_argument("--n_grid", type=int, required=False, default=300, help='the number of grid or h to devid the cell state space')
parser.add_argument("--n_dimension", type=int, required=False, default=5, help='the number of dimension to used for estimating density')
parser.add_argument("--timepoint_idx", type=str, required=False, default=None, help='the number of time point to train the model')
parser.add_argument("--resampling_indensity", type=float, required=False, default=0.5, help='the coefficient to scale the probability of a cell being sampled')
parser.add_argument("--resampling_rate", type=float, required=False, default=None, help='the rate of resampling over normal indexing')
parser.add_argument("--batch_size", type=int, required=False, default=128, help='the number of nearby cell state to include within a minibatch')
parser.add_argument("--tol", type=float, required=False, default=1e-4, help='the tolerance of error , used to control the precision and speed of ode integral')
parser.add_argument("--channels", type=str, required=False, default="3,32,32,1", help='the depth and width of the model')
parser.add_argument("--step_size", type=float, required=False, default=None, help='the step size used for ode int, default 0.005 for rk4 solver')
parser.add_argument("--D_penalty", type=float, required=False, default=None, help='the weight to regulate the level of D (Diffusion)')
parser.add_argument("--deltax_key", type=str, required=False, default="Delta_DM", help='the key to take deltax from adata')
parser.add_argument("--deltax_weight", type=float, required=False, default=1e-2, help='the weight used to regularize the similarity of deltax and v')
parser.add_argument("--weight_intensity", type=float, required=False, default=None, help='the weight to emphasize the high density cell, > 1 for weighting, <1 for unweighting')
parser.add_argument("--time_sensitive", action="store_true", required=False, help='Whether to include time in behavoir functions')

args = parser.parse_args()

# args = parser.parse_args([
#             "-D", "ery_mk", "-K", "DM_scaled",  "-M", "MLP_PINN", "-G", "6", "--n_timepoint", "5", 
#             "--batch_size", "50", "--channels", "6,32,32,1", "--n_dimension", "6", "--schedule_lr", "CyclicLR",
#             "--lr", "1e-4", "--n_timepoint" ,"7"])


                                ###               ###
                                #     read data     #
                                ###               ###

path = os.path.abspath(".")
h5_path = os.path.join(path, f'{args.dataset}.h5ad')
# find adata path
if not os.path.exists(h5_path):
    main_path = path
    h5_path = os.path.join(path, f'data/{args.dataset}.h5ad')
else:
    main_path = os.path.dirname(path)

adata = sc.read_h5ad(h5_path)

if args.timepoint_idx is None:
    timepoint_idx = adata.uns['pop']['t'][-1]
else:
    timepoint_idx = eval(args.timepoint_idx)



if args.log_name is None:
    logging_name = f"{args.dataset}-{args.cellstate_key}_n{timepoint_idx}"
else:
    logging_name = args.log_name
save_path = os.path.join(main_path, 'logs', logging_name, args.model+['','_tsense'][args.time_sensitive])
PINN.tl.make_dir(save_path)

# define model
model_class = eval(f"models.{args.model}")
channels = [int(c) for c in args.channels.split(",")]   

# for g v and D
if args.model == "pde_params": 
    n_dim = args.n_dimension + 1 if args.time_sensitive else args.n_dimension
    max_h = max(channels)
    model_kws = dict(v_channels = [n_dim, max_h, 32, args.n_dimension],
                    g_channels = [n_dim, max_h, 32, 1],
                    D_channels = [n_dim, max_h, 32, 1])
else:
    model_kws = {}

model = model_class(
        lr=3e-4,
        channels = channels,
        step_size = args.step_size,
        activation_fn='Tanh',
        ode_tol = args.tol,
        D_penalty = args.D_penalty, 
        deltax_weight = args.deltax_weight,
        weight_intensity = args.weight_intensity,
        **model_kws
    )

if args.pretrained is not None:
    Pretrain_class = args.pretrained.split("/")[2].replace("_tsense","")
    # Pretrain_class = "u_dt_weight"
    if Pretrain_class == args.model:
        model = model_class.load_from_checkpoint(args.pretrained, map_location='cpu')
    else:
        # then only u is use
        Pretrain_class = eval(f"models.{Pretrain_class}")
        Pretain_model = Pretrain_class.load_from_checkpoint(args.pretrained, map_location='cpu')
        # inherit the statedict
        state_dict = Pretain_model.model.state_dict()
        model.u.load_state_dict(state_dict)
        
# detecting pre-computed duds
duds_path = h5_path.replace(".h5ad", "_duds.npy")
if os.path.exists(duds_path):
    precomputed_duds = np.load(duds_path)
else:
    precomputed_duds = None

train_DS = reader.Duds_AnnDS(
                            AnnData=adata, 
                            timepoint_idx=timepoint_idx, 
                            precomputed_duds=precomputed_duds,
                            n_dimension = args.n_dimension,
                            cellstate_key=args.cellstate_key,  #'DM_EigenVector'
                            log_transform=False,
                            norm_time=False,
                            # resampling_indensity=args.resampling_indensity,
                            # resampling_rate=args.resampling_rate,
                            deltax_key=args.deltax_key,
                            batchsize=args.batch_size)

if precomputed_duds is None:
    np.save(duds_path, train_DS.duds)

def my_collection_fn(batch):

    s, (t, t_p1), (u_t, u_tp1) = batch

    s.require_grad_(True)
    t.require_grad_(True)
    t_p1.require_grad_(True)
    u_t.require_grad_(True)
    u_tp1.require_grad_(True)

    return s, (t, t_p1), (u_t, u_tp1)

train_DL = DataLoader(train_DS, batch_size=None, num_workers=10)
# train_iter = iter(train_DL)
# s, (t, t_p1), (u_t, u_tp1) = next(train_iter)



device = 'gpu' if torch.cuda.is_available() else 'cpu'
device = 'cpu' if args.gpu_devices == None else 'gpu'
gpu_device = args.gpu_devices

trainer = pl.Trainer(
                    #auto_lr_find=True,
                    accelerator=device,
                    # fast_dev_run=True,
                    # gradient_clip_val=0.5,
                    default_root_dir=save_path,
                    devices = [gpu_device], 
                    max_epochs=300,
                    callbacks=[callbacks.ModelCheckpoint(filename='{epoch}-{total_loss:.8f}',
                                                monitor="total_loss", mode="min", save_top_k=2)]
                    )

trainer.fit(model, train_dataloaders=train_DL)
