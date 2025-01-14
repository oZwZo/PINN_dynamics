import os, sys, re
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import time

from tqdm.auto import tqdm
from TorchDiffEqPack import odesolve
from torchdiffeq import odeint_adjoint as odeint
from scipy.stats import pearsonr, spearmanr
from scipy.special import kl_div
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
    
    u_b = DataSet.u_b.cpu().numpy().reshape(DataSet.T_b.shape[0], -1)

    if 'duds' in dir(DataSet):
        duds = DataSet.duds.copy()
    else:
        duds = np.zeros((u_b.shape[0], u_b.shape[1], cellstate.shape[1]))
    


    all_output = []
    chunk_size= 1000
    if timepoints[0] == 0:
        # mean minus
        t_list = timepoints
    else:
        t_list = timepoints/ timepoints[0] / pde_model.time_scale_factor

    for it, itp1 in tqdm(zip(timepoint_idx[:-1], timepoint_idx[1:])):

        out_t = []

        print("simulating from timepoint", t_list[it], "to", t_list[itp1])

        for i in range(0, len(cellstate), chunk_size):

            s0 = cellstate[i:i+chunk_size].to(device).requires_grad_()
            tompos_u0 = torch.from_numpy(u_b[it, i:i+chunk_size]).float().to(device).requires_grad_()
            duds_0 = torch.from_numpy(duds[it, i:i+chunk_size, :]).float().to(device).requires_grad_()
            
            y_0 = torch.zeros_like(tompos_u0)
            # init_condition = (tompos_u0, s0) if model_name == 'pde_params' else (tompos_u0, s0, duds_0, y_0.clone(), y_0.clone(), y_0.clone())
            init_condition = (tompos_u0, s0, duds_0, y_0.clone(), y_0.clone(), y_0.clone())

            int_out_raw = odeint(
                            pde_model,
                            y0 = init_condition,
                            t = torch.tensor(t_list[it:itp1+1]).type(torch.float32).to(device),
                            atol=pde_model.ode_tol,
                            rtol=pde_model.ode_tol,
                            method='dopri5',
                            adjoint_options={'norm':'seminorm'},
                        )
            int_out = []
            u_t = int_out_raw[0]
            if pde_model.log_transform:
                int_out.append(u_t[1:])
            else:
                int_out.append(torch.nn.functional.relu(u_t[1:]))

            del u_t

            int_out.extend(int_out_raw[1:])

            if len(out_t) == 0:
                out_t = [[] for i in range(len(int_out)) ]

            for i, o in enumerate(int_out):
                # add the i_th output
                out_t[i].append(o.detach().cpu().numpy())

            # torch.cuda.empty_cache()

        # concate at batch - wise

        for i, ith_out in enumerate(out_t):
            conate_axis = 1 if len(ith_out[0].shape) > 0 else 0 # 1 is sample wise if more than 1 evaluation timepoint
            ith_concate = np.concatenate(ith_out, axis=conate_axis)
            out_t[i] = ith_concate

       
        all_output.append(out_t)
            
        # it+=1

    # conate at timepoint-wise
    all_output = [np.concatenate([o[i] for o in all_output], axis=0) for i in range(len(out_t))]

    if return_all:
        return all_output
    else:
        return all_output[0]

def param_vs_score(adata, obs_key, param, timepoints=None,timepoint_key='timepoint_tx_days'):
    if timepoints is None:
        timepoints = adata.obs[timepoint_key].unique()

    assert len(timepoints) == param.shape[0], "the provided timepoints and the given params must the same dimension"
    spr = []
    pr = []
    score = adata.obs[obs_key].values
    for i,t in enumerate(timepoints):
        idx_t = adata.obs[timepoint_key] == t
        spr.append(spearmanr(param[i][idx_t], score[idx_t])[0] )
        pr.append( pearsonr(param[i][idx_t], score[idx_t])[0] )
    
    return np.array(spr), np.array(pr)

def W_distance(u_b, u_simulate, p=2, log_transform=False):
    r"""
    Normalize the density and compute the Wasserstein distance between observation and prediction.
    Log-density is supported, pass log_transform = True

    Input
    -------
    u_b : ndarray, [n_time, n_cell] , observed density
    u_simulate : ndarray, [n_time, n_cell], inferred desity
    p : int, degree of the distance, default W-2 distance
    sanity_check : bool, whether check shape and positivity

    Return 
    -------
    Wasserstein distance : ndarry, [n_time,]
    """

    log_transform = True if np.any(u_b<0) else log_transform

    distance_ls = []
    for t in range(u_b.shape[0]):

        if log_transform:
            u_b[t] = np.exp(u_b[t])
            u_simulate[t] = np.exp(u_simulate[t])

        # normalize
        p_b = u_b[t] / u_b[t].sum()
        p_int = u_simulate[t] / u_simulate[t].sum()

        if p==1:
            w = np.abs(p_b - p_int)
        elif p > 1:
            w = np.power(p_b - p_int, p)
            w = w**(1/p)
        distance_ls.append(np.sum(p_b*w))

    return np.array(distance_ls)

def KLD_density(u_b, u_simulate, sanity_check=True):
    r"""
    Normalize the density and compute the KL-divergence between observation and prediction

    Input
    -------
    u_b : ndarray, [n_time, n_cell] , observed density
    u_simulate : ndarray, [n_time, n_cell], inferred desity
    sanity_check : bool, whether check shape and positivity

    Return 
    -------
    KLD_ls : ndarray, [n_time, ]
    """
    if sanity_check:
        assert u_b.shape == u_simulate.shape, "observation and prediction must be the same"
        assert np.all(u_b>=0), "density must be positive"

    KLD_ls = []
    for t in range(u_b.shape[0]):

        # normalize
        p_b = u_b[t] / u_b[t].sum()
        p_sim = u_simulate[t] / u_simulate[t].sum()
        p_b += 1e-34
        p_sim += 1e-34

        # kld
        KLD_ls.append(kl_div(p_b, p_sim).sum())
    KLD_ls= np.array(KLD_ls)

    return KLD_ls