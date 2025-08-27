import os,sys,gc
import numpy as np
import pandas as pd
import torch
from torch import nn
import pytorch_lightning as pl
from typing import Any, Union
from typing import Any, Union, Callable
from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint 
# from TorchDiffEqPack import odesolve_adjoint_sym12
# from kan import KAN
import matplotlib.pyplot as plt

from ._pde_informed_params import *
from ._PINN_base import PINN_base, PINN_base_sim
from .MLP_models import MLP_surrogate
from .Spline_models import MultiDim_CubicSpline, CubicSpline
from ..functions import eval_funs as efun


class Density_Transfer(nn.Module):
    """
    Wrap up the density transfer function into a nn Module, for calling odeint_adjoint
    """
    def __init__(self, pde_model=None, stochastic=False, noise_schedule=None, n_repeat=30):
        super().__init__()
        self.model = pde_model
        self.relu = nn.ReLU()

        # for cell state drift 
        self.stochastic = stochastic
        self.n_repeat = n_repeat
        
        if noise_schedule is None:
            # the default noise schedule is a constant over time and state
            self.noise_schedule = lambda s, t: 1
        else:
            self.noise_schedule = noise_schedule
    
    def forward(self, t, states):
        return self.model.density_transfer(t, states)

    def velocity(self, t, s):
        device = s.device
        ncell  = s.shape[0]
        t_in = torch.full((s.shape[0],1), t.item()).to(device) *self.model.time_scale_factor

        if self.stochastic:
            noise =  torch.broadcast_to(torch.sqrt(self.model.D(s,t_in)*2).view(ncell,-1), s.shape)
            ds = self.model.v(s, t_in)  +  \
                self.noise_schedule(s, t_in) * noise * torch.randn((ncell,1)).to(device)
        else:
            # the state is deterministic and only decide by v
            ds = self.model.v(s, t_in)


        return ds

    def cellstate_drift(self, s0, integrate_time):
        
        # get device
        if isinstance(s0, np.ndarray):
            device = self.model.device
            s0 = torch.from_numpy(s0).float().requires_grad_().to(device)
        else:
            device = s0.device

        # integrate v
        with torch.no_grad():
            try:
                s_out = odeint(self.velocity, y0=s0, 
                        t=torch.tensor(integrate_time).to(device)*self.model.time_scale_factor,
                        atol=self.model.ode_tol,
                        rtol=self.model.ode_tol,

                        )
            except AssertionError:  #underflow
                step_size = np.around((integrate_time[-1] - integrate_time[0]) / 100 , 2)
                s_out = odeint(self.velocity, y0=s0, 
                        t=torch.tensor(integrate_time).to(device)*self.model.time_scale_factor,
                        atol=self.model.ode_tol,
                        rtol=self.model.ode_tol,
                        method='rk4',
                        options = {"step_size":step_size}
                        )
            
            s_traj = s_out.detach().cpu().numpy()
            del s_out 
        return s_traj
    
    def transition_by_batch(self, s0, u0, integrate_time, n_interval=10, ncell=200):
        """
        perform density transfer given initial cell state and density

        s0: tensor
        """
        # get device
        if isinstance(s0, np.ndarray):
            device = self.model.device
            s0 = torch.from_numpy(s0).float().requires_grad_().to(device)
        else:
            device = s0.device

        
        # step 1. cell state simulation
        s_traj_ay = self.cellstate_drift(s0, integrate_time)

        # step 2. density simulation
        u_traj_ay = []
        for ic in range(0, s0.shape[0], ncell*4):
            s0_chunk = s0[ic:ic+ncell*4]
            u0_chunk = u0[ic:ic+ncell*4]

            zeros = torch.zeros_like(u0_chunk)
            duds_init = torch.zeros_like(s0_chunk)

            intout  = odeint_adjoint(
                            self.model,
                            y0 = (u0_chunk, s0_chunk, duds_init, zeros, zeros, zeros),
                            t = torch.tensor(integrate_time).type(torch.float32).to(device),
                            atol=self.model.ode_tol,
                            rtol=self.model.ode_tol,
                            method='dopri5',
                        )
            u_traj_ay.append(self.relu(intout[0]).detach().cpu().numpy()) # (time, chunk)
            del intout

        if len(u_traj_ay)>1:
            u_traj_ay = np.concatenate(u_traj_ay, axis=1) # (time, cell)
        else:
            u_traj_ay = u_traj_ay[0]

        # step 3. density transfer
        Tmaps_t = []
        Tmaps_norm_t = []

        for ic in range(0, s0.shape[0], ncell):
            u0_chunk = u0[ic:ic+ncell]
            leftover_u_global = [ np.zeros((n_interval, u0_chunk.shape[0])) ]
            diff_flow_u_global = []

            # step 3.1
            for offset in range(n_interval):   # the ith time of transition

                s_traj = torch.from_numpy(s_traj_ay[:,ic:ic+ncell,:]).float().to(device)
                
                tn1 = integrate_time[0]
                diff_flow_u_local = [u_traj_ay[offset, ic:ic+ncell]]
                leftover_u_local = []

                # step 3.2
                for i, t in enumerate(integrate_time[1:n_interval-offset+1]):

                    s_tn1 = s_traj[i].requires_grad_().to(device)
                    s_t = s_traj[i+1].requires_grad_().to(device)
                    
                    left_u = torch.from_numpy(leftover_u_global[-1][i]).requires_grad_().to(device) # left from last round of diff trajactory
                    inflow_u = torch.from_numpy(diff_flow_u_local[-1]).requires_grad_().to(device)  # inflow from last step of the same round of diff trajectory
                    
                    # the 
                    u_tn1 = left_u.float() + inflow_u.float()
                    u_t = torch.zeros_like(u_tn1).to(device)

                    # integrate the flow within small timespan
                    s_last, u_stay, s_next, u_flow = odeint_adjoint(self, 
                                    y0= (s_tn1, u_tn1, s_t, u_t), 
                                    t = torch.tensor([tn1, t]).float().to(device),
                                    rtol = self.model.ode_tol,
                                    atol = self.model.ode_tol,
                                    )
                    
                    with torch.no_grad():
                        diff_flow_u_local.append( self.relu(u_flow[-1]).detach().to('cpu').numpy() )
                        leftover_u_local.append( self.relu(u_stay[-1]).detach().to('cpu').numpy() )
                        tn1 = t

                    del s_tn1, s_t, u_tn1, u_t, left_u, inflow_u, s_last, u_stay, s_next, u_flow
                    # free_memory(to_delete)
                    torch.cuda.empty_cache()
                    self.model.zero_grad()

                ##
                #  the nth offset, closing step 3.2

                diff_flow_u_local = np.stack(diff_flow_u_local)[1:] # length : 100 - offset
                leftover_u_local = np.stack(leftover_u_local)#[1:]   # length : 100 - offset 

                diff_flow_u_global.append(diff_flow_u_local)
                leftover_u_global.append(leftover_u_local)

                gc.collect()
                torch.cuda.empty_cache()

                del diff_flow_u_local, leftover_u_local,  s_traj #, self.model
                
            ##
            #  place the flow and leftover density into a triangle matrix
            #  closing step 3.1
            Tmap_manual = []
            for icell in range(u0_chunk.shape[0]):
                T_M = np.zeros((n_interval,n_interval))
                
                for i,flow_u_t in enumerate(diff_flow_u_global):
                    nt = flow_u_t.shape[0]
                    
                    for j, u in enumerate(flow_u_t[:,icell]):
                        T_M[i+j, j] = u

                Tmap_manual.append(T_M)


            Tmap = np.stack(Tmap_manual)
            Tmaps_t.append(Tmap)

            Tmap_norm = Tmap / u0_chunk.to('cpu').numpy().reshape(-1,1,1)
            Tmaps_norm_t.append(Tmap_norm)

        # closing step 3
        Tmaps_t_all = np.concatenate(Tmaps_t, axis=0)
        Tmaps_norm_t_all = np.concatenate(Tmaps_norm_t, axis=0)
        
        return Tmaps_t_all, Tmaps_norm_t_all


class DT_analysis:
    r"""
    class to analyze the density transport result from saved files
    """
    def __init__(self, adata, result_dir):
        self.adata = adata
        self.result_dir = result_dir
        self.load_result(result_dir)

    def load_result(self, result_dir):
        r"""
        load the result from saved files, this method returns 4 dictionary with timepoint as key
        
        Saved Properties:
        - cb_dict: a dictionary of cell barcode at its corresponding time point
        - TM_dict: a dictionary of *raw* transport map at its corresponding time point
        - TM_norm_dict: a dictionary of *normalized* transport map at its corresponding time point
        - trajectory_dict: a dictionary of trajectory at its corresponding time point
        """
        # load cell barcode dict
        
        

        # load density intermediate files
        self.TM_dict = {}                        # shape : [cell, step, step]
        self.TM_norm_dict = {}                   # shape : [cell, step, step]
        self.trajectory_dict = {}                # shape : [step+1, cell, n_dim ]
        self.cb_dict = {}
        ct_prop_ls = []

        # load trajectory and transport map
        npy_save_dir = os.path.join(self.result_dir, 'Density_transport')
        for files in os.listdir(npy_save_dir):

            day = files.split('_')[1].replace('Day', '')

            if files.endswith('cellbarcode.npy'):
                self.cb_dict[day] = np.load(os.path.join(npy_save_dir, files), allow_pickle=True)

            elif files.endswith('_Norm_TransportMap.npy'):
                self.TM_norm_dict[day] = np.load(os.path.join(npy_save_dir, files))
            
            elif files.endswith('_TransportMap.npy'):
                self.TM_dict[day] = np.load(os.path.join(npy_save_dir, files))

            elif files.endswith('sim_trajectory.npy'):
                self.trajectory_dict[day] = np.load(os.path.join(npy_save_dir, files))
            
            elif files.endswith('ct_prop.csv'):
                ct_prop = pd.read_csv(os.path.join(npy_save_dir, files))
                ct_prop_ls.append(ct_prop)
            else:
                print(f'{files} is not a valid file')
        
        if len(ct_prop_ls) >0:
            print("cell type proportion summary detected")
            print("adding to  ct_prop property")
            self.ct_prop = pd.concat(ct_prop_ls)

    def summarize_cell_proportions(self, df, celltype_list):
        """
        Summarizes the proportion of cell types mapped from neighbors for each cell

        Inputs:
        df : Input DataFrame where each column represents a cell and each row a sample.
        celltype_list : List of cell types to include in the output as columns.
        """
        # Compute normalized value counts for each cell
        proportions = df.apply(lambda col: col.value_counts(normalize=True))
        proportions = proportions.fillna(0).T # cell as index

        # Reindex columns to match the given celltype_list, filling missing with 0
        proportions = proportions.reindex(columns=celltype_list, fill_value=0)

        return proportions
    
    def annotate_trajectory(self, cellstate_key, obs_key, copy=False):
        r"""
        Use the nearest celltype to annotate each step along the simulated trajectory

        Inputs:
        cellstate_key: which cellstate space for the simulation to map to
        obs_key: the key to store the annotation in adata

        Returns:
        celltype_trajectory: with the annotation added
        """
        try:
            # the order matched with that in adata
            celltype_list = self.adata.obs[obs_key].cat.categories
        except:
            celltype_list = self.adata.obs[obs_key].unique()
        

        celltype_trajectory = {}

        for t, traj in self.trajectory_dict.items():

            # cb = self.cb_dict[t]
            
            ct_df = []
            for step in range(1, traj.shape[0]):

                annotations = efun.assign_nearest_cell(traj[step], self.adata, cellstate_key=cellstate_key, annotation=obs_key)
                proportions = self.summarize_cell_proportions(annotations, celltype_list)

                # assert proportions.shape[0] == len(cb)
                # proportions.index = cb

                proportions['Day'] = t
                proportions['step'] = step
                proportions['cell_index'] = proportions.index
                ct_df.append(proportions)
            
            celltype_trajectory[t] = pd.concat(ct_df).sort_values(['cell_index', 'Day'])
        
        if copy:
            self.celltype_trajectory = celltype_trajectory 
        else:
            return celltype_trajectory