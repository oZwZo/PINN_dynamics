import torch
import numpy as np
import pandas as pd
from . import functions as myfn
from torch.utils.data import Dataset, DataLoader, TensorDataset

 
class Pdyn_ExtractDataset(Dataset):
    
    def __init__(self, Data_pt, n_grid=300, collocation_points=600, n_repeat=10, log_transform=True):
        """
        PINN-dynamics Dataset using the pre-extracted data.   
        This dataset returns full cell state (0-1) for each mini-batch
        
        Augment
        --------
        Data_pt : the output file path from script
        """
        super().__init__()
        # load the result
        D = torch.load(Data_pt)
        
        if log_transform:
            mu = np.array(D['pop']['mean'])
            D['pop']['mean'] = np.log(mu)
            D['pop']['var'] = D['pop']['var']/ mu

            # if np.max(D['pop']['t']) >= 30:
            #     D['pop']['raw_t'] = D['pop']['t']
            #     D['pop']['t'] =  np.log(np.where(D['pop']['t']==0, 1, D['pop']['t']))
                
            #     D['ind']['tp'] = np.array(D['ind']['tp'])
            #     D['ind']['raw_tp'] = D['ind']['tp']
            #     D['ind']['tp'] = np.log(np.where(D['ind']['tp']==0, 1, D['ind']['tp']))
        ###
        # set up params
        ### 
        self.n_repeat = n_repeat
        self.N_coll = collocation_points

        ###
        # create grided cell state
        ###
        n_grid = n_grid
        self.n_grid = n_grid
        s = np.linspace(0, 1, n_grid)
        self.s = torch.from_numpy(s)
        h_inv = (1 / (s[1] - s[0]))
        
        ###
        # set up boundary conditions
        ### 
        ub_ls = []
        tb_ls = []
        var_ls = []
        
        for tb_idx, t_b in enumerate(D['pop']['t']):
            
            # quantify the cell densitied at grided s
            u, N, n_exp = myfn.boundary_density_at(D, t_b, s)
                    
            ub_ls.append(u / h_inv)
            tb_ls.append(np.full_like(u, t_b))
            var_ls.append(D['pop']['var'][tb_idx] /n_exp)
        
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


class Random_ExtractDataset(Pdyn_ExtractDataset):

    def __init__(self, Data_pt, nearby_cellstate=10, n_grid=300, collocation_points=600, n_repeat=10, log_transform=True):
        """
        Based on the pre-extracted dataset, 

        Augments
        -----------
        Data_pt :
        nearby_cellstate : the range of cell state in a minibatch

        """
        super().__init__(Data_pt, n_grid=n_grid, collocation_points=collocation_points, n_repeat=n_repeat, log_transform=log_transform)       
        self.nearby_cellstate = nearby_cellstate
    
    def __len__(self):
        # repeat sampling for 10 times
        return self.n_grid - self.nearby_cellstate

    def __getitem__(self, i):
        """
        there is no mini-batch, each item returns the full collocation points
        """ 

        s_range = slice(i, i + self.nearby_cellstate)


        # random collocation points across the s and t domain
        s_col = self.s.clone().detach().float().view(-1,1)[s_range,:]
        t_col = torch.Tensor(self.nearby_cellstate,1).uniform_(min(self.T_b), max(self.T_b)).float()
        
        
        t_b = torch.from_numpy(self.t_b).float()[:, s_range]
        s_all = self.s.clone().detach().float().view(1,-1)[:, s_range]
        s_all = s_all.broadcast_to(t_b.shape).float()

        #
        u_b = torch.from_numpy(self.u_b).float()[:, s_range]
        mean = 0.5*(u_b[:,1:]+u_b[:,:-1]).sum(dim=1, keepdim = True) #/ h_inv   

        # mean = torch.from_numpy(self.pop_mean).float()
        var = torch.from_numpy(self.pop_var).float()
        
        return s_col, t_col, s_all, t_b, u_b, mean, var



class MeshGrid_DS(Dataset):
    
    def __init__(self, Data_pt, nearby_cellstate=10, n_grid=300, collocation_points=600, n_repeat=10, log_transform=True, norm_time=True):
        """
        PINN-dynamics Dataset using the pre-extracted data.   
        This dataset returns full cell state (0-1) for each mini-batch
        
        Augment
        --------
        Data_pt : the output file path from script
        nearby_cellstate : the number of near (cell state)
        """
        super().__init__()
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
        self.nearby_cellstate = nearby_cellstate

        self.n_dim = D['ind']['hist'][0].shape[1]

        ###
        # create grided cell state
        ###
        self.n_grid = n_grid
        coords = [np.linspace(0.01, 0.99, n_grid) for i in range(self.n_dim)]  # generate 1D uniform coord
        meshgrid = np.vstack([ay.flatten() for ay in  np.meshgrid(*coords)]).T

        self.s = torch.from_numpy(meshgrid).float()
        h_inv = 1/np.prod([s[1] - s[0] for s in coords])

        if norm_time:
            T_b =  np.log(np.where(D['pop']['t']==0, 1, D['pop']['t']))
            T_b = T_b / T_b.max()
        
        ###
        # set up boundary conditions
        ### 
        ub_ls = []
        tb_ls = []
        var_ls = []
        
        for tb_idx, t_b in enumerate(D['pop']['t']):
            
            # quantify the cell densitied at grided s
            u, N, n_exp = myfn.boundary_density_at(D, t_b, meshgrid)
                    
            ub_ls.append(u / h_inv)
            tb_ls.append(np.full_like(u, T_b[tb_idx])) # add norm t
            var_ls.append(D['pop']['var'][tb_idx] /n_exp)
        
        self.u_b = np.vstack(ub_ls)  # (tb, n_grid)
        self.t_b = np.vstack(tb_ls)
        
        # observeds
        self.pop_var = np.array(var_ls)  # (tb,)
        self.pop_mean = D['pop']['mean'] # (tb,)
        self.T_b = D['pop']['t']         # (tb,)
        self.T_b = T_b

        

    def __len__(self):
        # repeat sampling for 10 times
        return self.s.shape[0] - self.nearby_cellstate

    def __getitem__(self, i):
        """
        there is no mini-batch, each item returns the full collocation points
        """ 

        s_range = slice(i, i + self.nearby_cellstate)

        # random collocation points across the s and t domain
        s_col = self.s.clone().detach().float()[s_range,:]
        t_col = torch.Tensor(self.nearby_cellstate,1).uniform_(min(self.T_b), max(self.T_b)).float()
        
        t_b = torch.from_numpy(self.t_b).float()[:, s_range].unsqueeze(-1)
        s_all = self.s.clone().detach().float()[s_range, :]

        bc_shape = [t_b.shape[0]] + list(s_all.shape)  # broadcast to
        s_all = s_all.broadcast_to(bc_shape).float()

        #
        u_b = torch.from_numpy(self.u_b).float()[:, s_range] + 1e-30
        # u_b = np.where(u_b==0, u_b.min(), u_b)
        mean = 0.5*(u_b[:,1:]+u_b[:,:-1]).sum(dim=1, keepdim = True) #/ h_inv   

        # mean = torch.from_numpy(self.pop_mean).float()
        var = torch.from_numpy(self.pop_var).float()


        # for nearby_cellstate == 1
        if self.nearby_cellstate == 1:
            s_all = s_all.squeeze(1)
            u_b = u_b.squeeze(1)
        
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