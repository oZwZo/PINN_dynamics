import torch
import numpy as np
import pandas as pd
from . import functions as myfn
from scipy.stats import gaussian_kde
from torch.utils.data import Dataset, DataLoader, TensorDataset

 
#TODO: complete AnnDataset
class AnnDataset(Dataset):
    
    def __init__(self, AnnData, cellstate_key='cellstate', timepoint_key='timepoint', pop_dict=None, n_grid=300, collocation_points=600,  log_transform=True):
        """
        PINN-dynamics Dataset, extract 

        Arguments:
        -----------
        AnnData : annData, the scanpy 
        cellstate_key : str, the obsm key, the lower dimension representation on which we will use to compute density
        timepoint_key : str, the obs key that indicate the experimental time the cells are collected from
        pop_dict : dict, the dictionary we use to pass population statistics including collected timepoint, mean ,variation

        Returns:
        ----------
        minibatch: not defined
        """
        super().__init__()
        self.adata = AnnData
        self.cellstate_key = cellstate_key
        self.timepoint_key = timepoint_key

        # check cell state
        assert cellstate_key in AnnData.obsm_keys(),  f'cellstate key `{cellstate_key}` not found in adata'
        assert timepoint_key in AnnData.obs_keys(),  f'timepoint key `{timepoint_key}` not found in adata'

        self.adata_tb = self.adata.obs[timepoint_key]          # pd.Series
        self.cellstate = self.adata.obsm[cellstate_key]        # np.values
        self.n_dim = self.cellstate.shape[1] # the dimension of the cell states

        # check poppulation
        if pop_dict is None:
            assert 'pop' in self.adata.uns, "please provide the cell population data"
            self.popD = self.adata.uns['pop'].copy()   # population Dict
        else:
            self.popD = pop_dict

        if log_transform:
            mu = np.array(self.popD['mean'])
            self.popD['mean'] = np.log(mu)
            self.popD['var'] = self.popD['var']/ mu

        ###
        # set up params
        ### 
        self.N_coll = collocation_points

        ###
        # create grided cell state
        ###
        n_grid = n_grid
        self.n_grid = n_grid


class OneBranch_AnnDS(AnnDataset):
    def __init__(self, *, n_repeat=10, **kwargs):
        """
        One branch system using one Dimensional cell state like pseudotime
        """
        super().__init__(**kwargs)
        self.n_repeat = n_repeat


class MeshGrid(Dataset):
    def __init__(self):
        super().__init__()
        self.s = None
        self.meshs = None
        self.n_grid = None
        self.nearby_cellstate =None
        self.u_b = None
        self.mesh_ub = None

    def __len__(self):
        # repeat sampling for 10 times
        return self.s.shape[0] - self.nearby_cellstate


    def resampling_by_density(self, n_samples, p=None):
        """
        sample meshes by the time-averaged density distribution
        """
        if p is None:

            try:  # detect density distribution 
                self.density_P
            except AttributeError:
                # create the distribution 
                ub_norm = self.u_b.sum(axis=1, keepdims=True)    # (t, n_grid**2)
                self.density_P = self.u_b/ub_norm

            p = np.mean(self.density_P, axis=0) # (n_grid**2, )

        
        # sample idx by their mean density over time

        a = np.arange(0, self.s.shape[0])
        idxs = np.random.choice(a, size=n_samples, p=p)
        return idxs
    
    def indexing_mesh(self, i):
        """
        given a sample index i in the flatten s, return the location in mesh-grid S
        """
        ix = i//self.n_grid
        iy = i%self.n_grid
        return ix, iy

    def indexing_flatten(self, ix, iy):
        """
        given a sample index i in the flatten s, return the location in mesh-grid S
        """
        return ix*self.n_grid + iy

    def indexing_neighbormesh(self, i, neighborhood=None):
        """
        looking for the index of neighbor mesh in a square , given bottom left index 1.
        
        """
        if neighborhood is None:
            neighborhood = self.nearby_cellstate
        
        bound = self.n_grid - self.nearby_cellstate
        
        squares = []
        ix, iy = self.indexing_mesh(i)   # locate square in mesh s 

        ix = ix if ix < bound else bound            # left bound
        iy = iy if iy < bound else bound            # bottom
        ix_end = min(ix+neighborhood, self.n_grid)  # right bound
        iy_end = min(iy+neighborhood, self.n_grid)  # up bound
        

        for ixx in range(ix, ix_end):
            for iyy in range(iy, iy_end):
                squares.append(self.indexing_flatten(ixx, iyy))
        return squares
        
        

    def __getitem__(self, i):
        """
        there is no mini-batch, each item returns the full collocation points
        """ 

        square_idx = self.indexing_neighbormesh(i, self.nearby_cellstate)
        # s_range = slice(i, i + self.nearby_cellstate)

        # random collocation points across the s and t domain
        s_col = torch.Tensor(self.N_coll,self.n_dim).uniform_(0.01, 0.99).float()
        t_col = torch.Tensor(self.N_coll,1).uniform_(min(self.T_b), max(self.T_b)).float()
        
        t_b = torch.from_numpy(self.t_b).float()[:, square_idx].unsqueeze(-1)
        s_bund = self.s.clone().detach().float()[square_idx, :]

        bc_shape = [t_b.shape[0]] + list(s_bund.shape)  # broadcast to
        s_bund = s_bund.broadcast_to(bc_shape).float()

        #
        u_b = torch.from_numpy(self.u_b[:, square_idx]).float()
        # u_b = np.where(u_b==0, u_b.min(), u_b)
        mean = 0.5*(u_b[:,1:]+u_b[:,:-1]).sum(dim=1, keepdim = True) #/ h_inv   

        # mean = torch.from_numpy(self.pop_mean).float()
        var = torch.from_numpy(self.pop_var).float()

        # for nearby_cellstate == 1
        if self.nearby_cellstate == 1:
            s_bund = s_bund.squeeze(1)
            u_b = u_b.squeeze(1)
        
        return s_col, t_col, s_bund, t_b, u_b, mean, var

class MeshGrid_AnnDS(AnnDataset, MeshGrid):
    def __init__(self, *, n_repeat=10, nearby_cellstate=10, norm_time=True, **kwargs):
        """
        Two branch system using mesh grid to span the all cell state space

        Augment
        --------
        n_repeat : the output file path from script
        nearby_cellstate : the number of near (cell state)
        norm_Time : log-normalize the real timepoint 
        """
        super().__init__(**kwargs)
        self.n_repeat = n_repeat
        self.nearby_cellstate = nearby_cellstate
    

        ###
        # create grided cell state
        ###
        coords = [np.linspace(0.01, 0.99, self.n_grid) for i in range(self.n_dim)]  # generate 1D uniform coord
        meshgrid_flat = np.vstack([ay.flatten() for ay in  np.meshgrid(*coords)]).T

        self.s = torch.from_numpy(meshgrid_flat).float()
        # self.meshs = self.s.reshape(self.n_grid,self.n_grid, -1) # from flatten to squared high-dim

        h_inv = 1/np.prod([s[1] - s[0] for s in coords])

        if norm_time:
            T_b =  np.log(np.where(self.popD['t']==0, 1, self.popD['t']))
            T_b = T_b / T_b.max()
        
        ###
        # set up boundary conditions
        ### 
        ub_ls = []
        tb_ls = []
        var_ls = []
        
        for tb_idx, t_b in enumerate(self.popD['t']):
            
            # subset ad_t
            
            cb_t = self.adata.obs.query(f"`{self.timepoint_key}` == @t_b").index
            ad_t = self.adata[cb_t].copy()
            cellstate_t = ad_t.obsm[self.cellstate_key]
            
            # assess density and return 
            density_fun = gaussian_kde(cellstate_t.T)
            u  = density_fun(meshgrid_flat.T)
            n_exp = self.popD['n_lib'][tb_idx]
                    
            ub_ls.append(u / h_inv * self.popD['mean'][tb_idx])
            tb_ls.append(np.full_like(u, T_b[tb_idx])) # add norm t
            var_ls.append(self.popD['var'][tb_idx] /n_exp)
        

        self.u_b = np.vstack(ub_ls) + 1e-30  # (tb, n_grid**2)
        # self.mesh_ub = self.u_b.reshape(-1, self.n_grid,self.n_grid) # (tb, n_grid, n_grid)
        self.t_b = np.vstack(tb_ls)

        # norm_p
        ub_norm = self.u_b.sum(axis=1, keepdims=True)    # (t, n_grid**2)
        self.density_P = self.u_b/ub_norm
        
        # observeds
        self.pop_var = np.array(var_ls)  # (tb,)
        self.pop_mean = self.popD['mean'] # (tb,)
        self.T_b = self.popD['t']         # (tb,)
        self.T_b = T_b    
    

class MeshGrid_Resample(MeshGrid_AnnDS):
    def __init__(self, *args, **kwargs):
        """
        the MeshGrid dataset with resampling trick
        """
        super().__init__(*args, **kwargs)

    def __len__(self):
        # repeat sampling for 10 times
        return self.s.shape[0] * (len(self.T_b)  + 1)

    def __getitem__(self, i):
        
        tb_i = i // self.s.shape[0] - 1
        tb_p = self.density_P[tb_i]

        if tb_i >= 0:
            resampled_i = self.resampling_by_density(1,p=self.density_P[tb_i]).item()
        else:
            resampled_i = i

        return super().__getitem__(resampled_i)

class MeshGrid_logDS(MeshGrid_Resample):
    def __init__(self, *args, **kwargs):
        """
        Two branch system using mesh grid to span the all cell state space

        Augment
        --------
        n_repeat : the output file path from script
        nearby_cellstate : the number of near (cell state)
        norm_Time : log-normalize the real timepoint 
        """
        super().__init__(*args, **kwargs)
        self.u_b = np.log(self.u_b[:4])

        # self.mesh_ub = self.u_b.reshape(-1, self.n_grid,self.n_grid) # (tb, n_grid, n_grid)
        self.t_b = self.t_b[:4]

        # norm_p
        ub_norm = self.u_b.sum(axis=1, keepdims=True)    # (t, n_grid**2)
        self.density_P = self.u_b/ub_norm
        
        # observeds
        self.pop_var = self.pop_var[:4]  # (tb,)
        self.pop_mean = self.pop_mean[:4] # (tb,)
        self.T_b = self.T_b[:4]         # (tb,)

class Simple_DS(MeshGrid_AnnDS):
    
    def __init__(self, *, n_timepoint, **kwargs):
        """
        only use the a specific time point
        """
        super().__init__(**kwargs)

        self.spec_t = self.T_b[:n_timepoint]

        self.u_b = torch.from_numpy(self.u_b[:n_timepoint].flatten()).float()
        # self.u_b = torch.log(self.u_b)
        self.t_b = torch.from_numpy(self.t_b[:n_timepoint].flatten().reshape(-1,1)).float()
        self.s = torch.concat([self.s]*len(range(0, n_timepoint)), dim=0).float()
        scaled_P = self.density_P[:n_timepoint].flatten() ** 0.8
        self.density_P = scaled_P / scaled_P.sum()

    def __len__(self):
        return self.u_b.shape[0]

    def __getitem__(self, i):

        i = self.resampling_by_density(1, p=self.density_P)

        return self.s[i], self.t_b[i], self.u_b[i]


class Processed_baseDS(Dataset):
    
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

class Pdyn_ExtractDataset(Processed_baseDS):
    
    def __init__(self, Data_pt, n_grid=300, collocation_points=600, n_repeat=10, log_transform=True):
        """
        PINN-dynamics Dataset using the pre-extracted data.   
        This dataset returns full cell state (0-1) for each mini-batch
        
        Augments
        --------
        Data_pt : the output file path from script

        Returns:
        ----------
        s_col : tensor, collocation cellstate coords
        t_col : tensor, collocation time point
        s_bund : tensor, the boundary cellstate coords
        t_b : tensor, the observed boundary time point
        u_b : tensor, the observed density at each time point, evaluated at the s_bund
        mean : tensor, the mean population size of the cell
        var
        """
        super().__init__(Data_pt, n_grid=n_grid, collocation_points=collocation_points, n_repeat=n_repeat, log_transform=log_transform)
        # load the result
        D = torch.load(Data_pt)

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
        s_bund = self.s.clone().detach()
        s_bund = s_bund.broadcast_to(t_b.shape).float()

        #
        u_b = torch.from_numpy(self.u_b).float()
        mean = torch.from_numpy(self.pop_mean).float()
        var = torch.from_numpy(self.pop_var).float()
        
        return s_col, t_col, s_bund, t_b, u_b, mean, var


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
        s_bund = self.s.clone().detach().float().view(1,-1)[:, s_range]
        s_bund = s_bund.broadcast_to(t_b.shape).float()

        #
        u_b = torch.from_numpy(self.u_b).float()[:, s_range]
        mean = 0.5*(u_b[:,1:]+u_b[:,:-1]).sum(dim=1, keepdim = True) #/ h_inv   

        # mean = torch.from_numpy(self.pop_mean).float()
        var = torch.from_numpy(self.pop_var).float()
        
        return s_col, t_col, s_bund, t_b, u_b, mean, var



class MeshGrid_DS(Processed_baseDS, MeshGrid):
    
    def __init__(self, Data_pt, nearby_cellstate=10, n_grid=300, collocation_points=600, n_repeat=10, log_transform=True, norm_time=True):
        """
        PINN-dynamics Dataset using the pre-extracted data.   
        This dataset returns full cell state (0-1) for each mini-batch
        
        Augment
        --------
        Data_pt : the output file path from script
        nearby_cellstate : the number of near (cell state)
        """
        super().__init__(Data_pt, n_grid=n_grid, collocation_points=collocation_points, n_repeat=n_repeat, log_transform=log_transform)
        # load the result
        D = torch.load(Data_pt)
        self.n_dim = D['ind']['hist'][0].shape[1]

        ###
        # create grided cell state
        ###
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