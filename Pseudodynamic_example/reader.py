import torch
import numpy as np
import pandas as pd
import functions as myfn
from torch.utils.data import Dataset, DataLoader, TensorDataset


class Pdyn_ExtractDataset(Dataset):
    
    def __init__(self, Data_pt, n_grid=300, collocation_points=600, n_repeat=10, log_transform=True):
        """
        PINN-dynamics Dataset with the output of extracted data. The  
        
        Augment
        --------
        Data_pt : the output file path from script
        """
        # load the result
        D = torch.load(Data_pt)
        
        if log_transform:
            mu = np.array(D['pop']['mean'])
            D['pop']['mean'] = np.log(mu)
            D['pop']['var'] = D['pop']['var']/ mu

        ###
        # set up params
        ### 
        self.n_repeat = n_repeat
        self.N_coll = collocation_points

        ###
        # create grided cell state
        ###
        n_grid = n_grid
        s = np.linspace(0, 1, n_grid)
        self.s = torch.from_numpy(s)
        h_inv = (1 / (s[1] - s[0]))
        
        ###
        # set up boundary conditions
        ### 
        ub_ls = []
        tb_ls = []
        var_ls = []
        
        for t in D['pop']['t']:
            
            # quantify the cell densitied at grided s
            u, N, n_exp = myfn.boundary_density_at(D, t, s)
                    
            ub_ls.append(u / h_inv)
            tb_ls.append(np.full_like(u, t))
            var_ls.append(D['pop']['var'][t] /n_exp)
        
        self.u_b = np.vstack(ub_ls)  # (tb, n_grid)
        self.t_b = np.vstack(tb_ls)
        
        # observeds
        self.pop_var = np.array(var_ls)  # (tb,)
        self.pop_mean = D['pop']['mean'] # (tb,)
        self.T_b = D['pop']['t']         # (tb,)
        
        
    def __len__(self):
        return self.n_repeat
    
        
    def __getitem__(self, i):
        """
        there is no mini-batch, each item returns the full collocation points
        """ 
        
        # random collocation points across the s and t domain
        s_col = torch.Tensor(self.N_coll,1).uniform_(0, 1).float()
        t_col = torch.Tensor(self.N_coll,1).uniform_(min(self.T_b), max(self.T_b)).float()
        
        
        t_b = torch.from_numpy(self.t_b).float()
        s_all = self.s.clone().detach()
        s_all = s_all.broadcast_to(t_b.shape).float()

        #
        u_b = torch.from_numpy(self.u_b).float()
        mean = torch.from_numpy(self.pop_mean).float()
        var = torch.from_numpy(self.pop_var).float()
        
        return s_col, t_col, s_all, t_b, u_b, mean, var
        
        
#TODO: complete AnnDataset
class Pdyn_AnnDataset(Dataset):
    
    def __init__(self, AnnData, Meta):
        """
        PINN-dynamics Dataset
        """
        
    def __len__(self):
        raise NotImplementederror
    
    def __getitem(self, i):
        raise NotImplementederror