import os, sys, re
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import time

from tqdm.auto import tqdm
from TorchDiffEqPack import odesolve
from torchdiffeq import odeint_adjoint as odeint

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.patches import Patch

def savefig(fig, name, path, dpi=200):
    fig.savefig(f'{path}/{name}.png', transparent=True, bbox_inches='tight', dpi=dpi)

def forward_get_params(pde_model, DataSet, t_ts=None, s_ts=None):
    r"""
    Given the setup dataset, evalute the behavior functions

    Arguments:
    ----------
    pde_model : nn.Module, sub-class of PINN.models.pde_params_base
    DataSet : sub-class of `PINN.readers.HighdimAnnDS` 
    t_ts : None , time point tensor
    s_ts : None , cell state tensor
    """
    device = pde_model.device
    timepoint_label = DataSet.popD['t']

    # FORWARD 
    u_pred_ls = []
    v_ls = []
    D_ls = []
    g_ls = []
    chunk_size= 500

    s_ts = DataSet.s.float().to(device) if s_ts is None else s_ts

    t_ts = DataSet.t_b.float().to(device) if t_ts is None else t_ts

    with torch.no_grad():
        for i in tqdm(range(0, len(t_ts), chunk_size)):                              
            s_in = s_ts[i:i+chunk_size]
            t_in = t_ts[i:i+chunk_size]

            v_pred = pde_model.v(s_in, t_in)
            g_pred = pde_model.g(s_in, t_in)
            D_pred = pde_model.D(s_in, t_in)
            
            v_ls.append(v_pred.detach().cpu().numpy())
            g_ls.append(g_pred.detach().cpu().numpy())
            D_ls.append(D_pred.detach().cpu().numpy())

            if "u" in dir(pde_model):
                u_pred = torch.exp(pde_model.u(s_in, t_in))
                u_pred_ls.append(u_pred.detach().cpu().numpy())

        
    # # other params
    n_dimension = s_in.shape[1]
    n_timepoint = len(timepoint_label)

    # concate
    g_pred_ay = np.concatenate(g_ls, axis=0).reshape(n_timepoint,-1)
    v_pred_ay = np.concatenate(v_ls, axis=0).reshape(n_timepoint, -1, n_dimension)
    D_pred_ay = np.concatenate(D_ls, axis=0).reshape(n_timepoint,-1)


    return u_pred_ls, g_pred_ay, v_pred_ay, D_pred_ay

def density_shortterm_simulation(pde_model, DataSet, timepoint_idx=None, time_span=1, timepoints=None, cellstate=None, return_all=False):
    r"""
    simulate density for each cells for any two consecutive timepoints

    Arguments:
    ----------
    pde_model : nn.Module, sub-class of PINN.models.pde_params_base
    DataSet : sub-class of `PINN.readers.HighdimAnnDS` 
    timepoint_idx : list of index , default the full timepoints defined in DataSet
    time_span : int , how many step of the timepoint index
    return_all : bool, if return the other output
    
    Return:
    ---------
    u_int_all : np.ndarry [n_timepoints, n_cells]
    """

    device = pde_model.device
    model_name = type(pde_model).__name__

    if timepoints is None:
        timepoints = DataSet.popD['t']

    if cellstate is None:
        cellstate = torch.from_numpy(DataSet.cellstate).float()

    if timepoint_idx is None:
        timepoint_idx = np.arange(len(timepoints))
    
    duds = DataSet.duds.copy()
    u_b = DataSet.u_b.cpu().numpy().reshape(DataSet.T_b.shape[0], -1)


    all_output = []
    chunk_size= 1000
    t_list = timepoints/ timepoints[0] / pde_model.time_scale_factor

    for it, itp1 in tqdm(zip(timepoint_idx[:-1], timepoint_idx[1:])):

        out_t = []

        for i in range(0, len(cellstate), chunk_size):

            s0 = cellstate[i:i+chunk_size].to(device).requires_grad_()
            tompos_u0 = torch.from_numpy(u_b[it, i:i+chunk_size]).float().to(device).requires_grad_()
            duds_0 = torch.from_numpy(duds[it, i:i+chunk_size, :]).float().to(device).requires_grad_()
            
            init_condition = (tompos_u0, s0) if model_name == 'pde_params' else (tompos_u0, s0, duds_0)

            int_out = odeint(
                            pde_model,
                            y0 = init_condition,
                            t = torch.tensor(t_list[it:itp1+1]).type(torch.float32).to(device),
                            atol=pde_model.ode_tol,
                            rtol=pde_model.ode_tol,
                            method='dopri5',
                            adjoint_options={'norm':'seminorm'},
                        )

            u_t = int_out[0]

            int_out[0] = torch.nn.functional.relu(u_t[1:])

            del u_t, s_t

            if return_all:
                if len(out_t) == 0:
                    out_t = [[] for i in range(len(int_out)) ]

                for i, o in enumerate(int_out):
                    # add the i_th output
                    out_t[i].append(o.detach().cpu().numpy())

            # torch.cuda.empty_cache()

        # concate at batch - wise
        if return_all:
            for i, ith_out in enumerate(out_t):
                conate_axis = 1 if len(ith_out[0].shape) > 0 else 0 # 1 is sample wise if more than 1 evaluation timepoint
                ith_concate = np.concatenate(ith_out, axis=conate_axis)
                out_t[i] = ith_concate

        u_int = np.concatenate(u_t_ls, axis=len(u_int.shape)-1)
        
        u_int_all_ls.append(u_int)
        if return_all:
            all_output.append(out_t)
            
        # it+=1

    # conate at timepoint-wise

    all_output

    if return_all:
        return all_output
    else:
        return u_int_all

