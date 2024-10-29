import torch
import numpy as np
import pandas as pd
from . import functions as tl
from scipy.stats import gaussian_kde
from torch.utils.data import Dataset, DataLoader, TensorDataset
from ._base_Dataset import AnnDataset, MeshGrid, Processed_baseDS
# from Dataset_base 
 
                                ##################################
                                ## Trajectory Independent  DS   ##
                                ##################################



class HigDim_AnnDS(AnnDataset):
    def __init__(self, *, n_timepoint=None, n_dimension=5, nearby_cellstate=1, norm_time=False, kde_kws={},**kwargs):
        r"""
        High Dimensional Cell state Dataset for trajectory indepdent modeling

        Augment
        --------
        n_repeat : the output file path from script
        nearby_cellstate : the number of near (cell state)
        norm_Time : log-normalize the real timepoint 

        Other params from AnnDataset:
        --------
        AnnData : annData, the scanpy 
        cellstate_key : str, the obsm key, the lower dimension representation on which we will use to compute density
        timepoint_key : str, the obs key that indicate the experimental time the cells are collected from
        pop_dict : dict, the dictionary we use to pass population statistics including collected timepoint, mean ,variation
        log_transform : bool, default False, whether the population size will be log transformed to reduce the magnitude of the data

        """
        super().__init__(**kwargs)
        
        self.n_dimension = n_dimension
        self.nearby_cellstate = nearby_cellstate
        self.n_timepoint = n_timepoint
        self.popD['t'] = self.popD['t'][:n_timepoint]

        # subset the adata
        if n_timepoint is not None:
            t_max = self.popD['t'].max()
            cbs = self.adata.obs.query(f"`{self.timepoint_key}` <= @t_max").index
            self.adata = self.adata[cbs]

        # use the cell state key of the entire dataset 
        # as it tells what are the possible points of the entire cell state space 
        cellstate = self.adata.obsm[self.cellstate_key][:, :n_dimension]
        self.cellstate = cellstate

        self.s = torch.from_numpy(cellstate).float()
        self.s = torch.cat([self.s]*len(self.popD['t'])).float()
    

        if norm_time:
            T_b =  np.log(np.where(self.popD['t']==0, 1, self.popD['t']))
            T_b = T_b / T_b.max()
        else:
            T_b = self.popD['t']
            T_b = T_b / T_b.min() 
        
        ###
        # set up boundary conditions 
        # compute the densities
        ### 
        ub_ls = []
        tb_ls = []
        var_ls = []
        density_funs = []
        cb_ls = []

        for tb_idx, t_b in enumerate(self.popD['t']):
            
            # subset ad_t
            
            cb_t = self.adata.obs.query(f"`{self.timepoint_key}` == @t_b").index
            ad_t = self.adata[cb_t].copy()
            cellstate_t = ad_t.obsm[self.cellstate_key][:, :n_dimension]
            
            # assess density and return 
            density_fun = gaussian_kde(cellstate_t.T, **kde_kws)
            u  = density_fun(cellstate.T)   # evaluate with the entire space
            n_exp = self.popD['n_lib'][tb_idx]

            # u_min = np.min(u[u!=0])
            u = np.where(u!=0, u, 1e-10) # replace 0 with 0.1* u_min
            u = u / u.sum()
            u = np.clip(u, a_min=1e-10, a_max=None) 
                    
            ub_ls.append(u * self.popD['mean'][tb_idx]) # TODO: check what are the sum of the density
            tb_ls.append(np.full_like(u, T_b[tb_idx])) # add norm t
            var_ls.append(self.popD['var'][tb_idx] /n_exp)
            cb_ls.append(cb_t)
            density_funs.append(density_fun)

        self.u_b = np.vstack(ub_ls)   # (tb, n_cell)
        self.t_b = torch.from_numpy(np.vstack(tb_ls).flatten()).float()
        self.density_funs = density_funs

        self.cb_ls = np.concatenate(cb_ls)
        # self.adata = self.adata[cb_ls].copy()

        # norm_p
        ub_norm = self.u_b.sum(axis=1, keepdims=True)    # (t, n_cell)
        self.density_P = self.u_b/ub_norm                #TODO:check shape and the values
        scaled_P = self.density_P.flatten() ** self.resampling_indensity
        self.density_P = scaled_P / scaled_P.sum() 
        
        self.u_b = torch.from_numpy(self.u_b.flatten()).float()

        
        # observeds
        self.pop_var = np.array(var_ls)  # (tb,)
        self.pop_mean = self.popD['mean'] # (tb,)
        self.T_b = T_b         # (tb,)

        self.s_std = torch.from_numpy(self.cellstate.std(axis=0)).float()

    def __len__(self):
        return self.s.shape[0] 

    def __getitem__(self, i):
        
        # boundary points
        # resampling rate  is set to 0.5
        if np.random.random() <= self.resampling_rate: 
            i = self.resampling_by_density(1, p=self.density_P).item()
        s_bon = self.s[i]      
        t_bon = self.t_b[i]
        u_bon = self.u_b[i]

        # collocalization point
        err = np.random.randn()
        i_col = np.random.choice(range(len(self.cellstate)))
        s_col = self.s[i] + err * self.s_std
        # s_col = torch.from_numpy(s_col).float()

        t_col = np.random.uniform(self.t_b.min().item(), self.t_b.max().item())
        t_col = torch.tensor([t_col]).float()

        return  s_col, t_col, s_bon.squeeze(), t_bon, u_bon


class TwoTimpepoint_AnnDS(HigDim_AnnDS):
    def __init__(self, *args, batchsize=200, **kwargs):
        r"""
        Dataset for high dimensional cellstate
        Each batch returns the cellstates, and their density in two consecutive timepoints
        
        Augments
        --------
        n_repeat : the output file path from script
        nearby_cellstate : the number of near (cell state)
        norm_Time : log-normalize the real timepoint 

        Other params from AnnDataset:
        --------
        AnnData : annData, the scanpy 
        cellstate_key : str, the obsm key, the lower dimension representation on which we will use to compute density
        timepoint_key : str, the obs key that indicate the experimental time the cells are collected from
        pop_dict : dict, the dictionary we use to pass population statistics including collected timepoint, mean ,variation
        log_transform : bool, default True, whether the population size will be log transformed to reduce the magnitude of the data
        """
        super().__init__(*args,**kwargs)
        self.batchsize = batchsize

        self.u_b = self.u_b.reshape(self.n_timepoint, -1)
    
    def __len__(self):
        return int(self.cellstate.shape[0] // self.batchsize) * 9

    def __getitem__(self, i):
        

        # sample current t
        i_t = np.random.randint(0, self.n_timepoint-1)  # the i^th timepoint index
        i_tp1 = i_t + 1                                 # the index of next timepoint
            
        # sample cellstates
        s_index = np.random.choice(np.arange(self.cellstate.shape[0]), size=(self.batchsize,), replace=False)
        s = torch.from_numpy(self.cellstate[s_index]).float()

        # get time
        t = torch.full(size=(self.batchsize,), fill_value=self.T_b[i_t]).float()
        t_p1 = torch.full(size=(self.batchsize,), fill_value=self.T_b[i_tp1]).float()
        

        # the density of two consecutive 
        u_t = self.u_b[i_t, s_index]
        u_tp1 = self.u_b[i_tp1, s_index]  # density of the t plus 1

        return  s, (t, t_p1), (u_t, u_tp1)
    
                                ################################
                                ## Trajectory Dependent  DS   ##
                                ################################
class SingleBranch_AnnDS(AnnDataset, MeshGrid):
    def __init__(self, *,n_timepoint=None, n_repeat=10, nearby_cellstate=10, max_timespan = 3, replicate_key = 'batch', **kwargs):
        """
        Single branch system using pseudotime grid to span the all cell state space

        Augment
        --------
        n_repeat : the output file path from script
        nearby_cellstate : the number of near (cell state)
        norm_Time : log-normalize the real timepoint 
        """
        super().__init__(**kwargs)
        self.n_repeat = n_repeat
        self.nearby_cellstate = nearby_cellstate
        self.h_inv = 1/self.n_grid
        self.replicate_key = replicate_key

        self.popD['t'] = self.popD['t'][:n_timepoint]
        self.n_timepoint = len(self.popD['t'])

        coords = np.linspace(0.01, 0.99, self.n_grid) # generate 1D uniform coord
        self.s = coords
        self.grid_cellstate = coords

        # density
        self.cellstate = self.cellstate.flatten()
        ub_ls, hist_var_ls,area_var_ls, tb_ls, var_ls = self.compute_grid_density() # this is from MeshGrid

        self.u_b = np.vstack(ub_ls) + 1e-30  # (tb, n_grid)
        # self.mesh_ub = self.u_b.reshape(-1, self.n_grid,self.n_grid) # (tb, n_grid, n_grid)
        self.hist_var = np.vstack(hist_var_ls)
        self.area_var = np.vstack(area_var_ls)
        self.t_b = np.vstack(tb_ls)
        
        # observeds
        self.pop_mean = self.popD['mean'] # (tb,)
        self.pop_var = np.array(var_ls)  # (tb,)

        # timepoint pair:
        self.timpoint_pairs_index = []
        
        if max_timespan is None:
            max_timespan = self.n_timepoint
        else:
            if max_timespan<=0:
                raise ValueError("max time span must larger than 0")
            max_timespan = max_timespan + 1

        for span in range(1,max_timespan):  # [1, N_t -1]
            for start in range(self.n_timepoint-span):
                self.timpoint_pairs_index.append( [start, start+span] )


    def __len__(self):
        return len(self.timpoint_pairs_index)

    def __getitem__(self, i):
        # no resampling

        indexs = torch.Tensor(self.timpoint_pairs_index[i]).long()

        s = torch.from_numpy(self.s).clone().float()
        t_b = torch.from_numpy(self.T_b).float()
        u_b = torch.from_numpy(self.u_b).float()
        hist_var = torch.from_numpy(self.hist_var).float()
        area_var = torch.from_numpy(self.area_var).float()

        mean = torch.from_numpy(self.pop_mean).float()
        var = torch.from_numpy(self.pop_var).float()
        return s, t_b, u_b, hist_var, area_var, mean, var, indexs

class MeshGrid_AnnDS(AnnDataset, MeshGrid):
    def __init__(self, *,n_timepoint=None, n_repeat=10, nearby_cellstate=10, norm_time=True, replicate_key='batch', **kwargs):
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
        self.h = 1/self.n_grid
        self.replicate_key = replicate_key

        self.n_timepoint = n_timepoint
        self.popD['t'] = self.popD['t'][:n_timepoint]

        # subset the adata
        if n_timepoint is not None:
            t_max = self.popD['t'].max()
            cbs = self.adata.obs.query(f"`{self.timepoint_key}` <= @t_max").index
            self.adata = self.adata[cbs]

        ###
        # create grided cell state
        ###
        coords = [np.linspace(0.01, 0.99, self.n_grid) for i in range(self.n_dim)]  # generate 1D uniform coord
        meshgrid_flat = np.vstack([ay.flatten() for ay in  np.meshgrid(*coords)]).T

        self.s = meshgrid_flat
        self.grid_cellstate = meshgrid_flat.T
        # self.meshs = self.s.reshape(self.n_grid,self.n_grid, -1) # from flatten to squared high-dim

        self.h_inv = 1/np.prod([s[1] - s[0] for s in coords])
        
        ###
        # set up boundary conditions
        ### 
        ub_ls, hist_var_ls,area_var_ls, tb_ls, var_ls  = self.compute_grid_density()
        

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

class TwoTimepoint_MeshGrid(MeshGrid_Resample):
    def __init__(self, *args, **kwargs):
        """
        the MeshGrid dataset returning the data of two consecutive timepoints
        """
        super().__init__(*args, **kwargs)

    def __len__(self):
        return self.s.shape[0] 

    def __getitem__(self, i):
        
        # sample current t
        i_t = np.random.randint(0, self.n_timepoint-1)  # the i^th timepoint index
        i_tp1 = i_t + 1                                 # the index of next timepoint

        if np.random.random() <= self.resampling_rate: 
            i = self.resampling_by_density(1, p=self.density_P[i_t]).item()
    
            
        # sample cellstates
        s = self.s[i].float()

        # get time
        t = torch.tensor([self.T_b[i_t]]).float()
        t_p1 = torch.tensor([self.T_b[i_tp1]]).float()
        

        # the density of two consecutive 
        u_t = torch.from_numpy(self.u_b[i_t, [i]]).float()
        u_tp1 = torch.from_numpy(self.u_b[i_tp1, [i]]).float()  # density of the t plus 1

        return  s, (t, t_p1), (u_t, u_tp1)

class AllTimepoint_MeshGrid(MeshGrid_Resample):
    def __init__(self, *args, **kwargs):
        """
        the MeshGrid dataset returning the data of two consecutive timepoints
        """
        super().__init__(*args, **kwargs)

    def __len__(self):
        return self.s.shape[0] 

    def __getitem__(self, i):
        

        if np.random.random() <= self.resampling_rate: 
            tb_i = np.random.choice(range(self.density_P.shape[0]))
            i = self.resampling_by_density(1, p=self.density_P[tb_i]).item()

        s_bund = self.s[i,:].copy()
        s_bund = torch.from_numpy(s_bund).float()
        t_b = torch.from_numpy(self.t_b[:, i]).float().unsqueeze(-1)
        u_b = torch.from_numpy(self.u_b[:, i]).float()

        squre_indexes = self.indexing_neighbormesh_center(i, neighborhood=3)

        s_neighbor = self.s[squre_indexes].reshape(3,3,2)
        s_neighbor = torch.from_numpy(s_neighbor).float()
        u_neighbor = torch.from_numpy(self.u_b[:,squre_indexes]).reshape(-1, 3,3).float()
        return (s_bund,s_neighbor), t_b, (u_b, u_neighbor)


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

                                ########################
                                ##     Sim DataSet    ##
                                ########################

class Simple_DS(MeshGrid_AnnDS):
    
    def __init__(self, *, n_timepoint, **kwargs):
        """
        only use the a specific time point
        """
        super().__init__(**kwargs) 

        self.T_b = self.T_b[:n_timepoint]

        self.u_b = torch.from_numpy(self.u_b[:n_timepoint].flatten()).float()
        # self.u_b = torch.log(self.u_b)
        self.t_b = torch.from_numpy(self.t_b[:n_timepoint].flatten().reshape(-1,1)).float()
        self.s = torch.concat([self.s]*len(range(0, n_timepoint)), dim=0).float()
        scaled_P = self.density_P[:n_timepoint].flatten() ** 0.5
        self.density_P = scaled_P / scaled_P.sum() 

        for key in self.popD:
            self.popD[key] = self.popD[key][:n_timepoint]

    def __len__(self):
        return self.u_b.shape[0]

    def __getitem__(self, i):

        i = self.resampling_by_density(1, p=self.density_P)

        return self.s[i], self.t_b[i], self.u_b[i]


                                ########################
                                ##  Processed DataSet ##
                                ########################


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
            u, N, n_exp = tl.boundary_density_at(D, t_b, s)
                    
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
            u, N, n_exp = tl.boundary_density_at(D, t_b, meshgrid)
                    
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