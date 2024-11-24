import torch
import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm
from scipy.stats import gaussian_kde,entropy
from scipy.integrate import trapz
from . import models


def scale_dpt(dpt):
    """
    scale dpt array to 0 and 1
    """
    dpt_min = dpt.min()
    dpt_max = dpt.max()
    dpt_scaled = (dpt - dpt_min) / (dpt_max - dpt_min)
    
    return dpt_scaled

def load_model(ckptvx):
    r"""
    from a given ckpt , load the MLP model
    """
    model_vx = models.MLP(lr=1e-4, channels=[6,32,32,1], activation_fn='Tanh')
    ckpt = torch.load(ckptvx)
    model_vx.load_state_dict(ckpt['state_dict'])
    return model_vx

def pred_to_nday(x, n_timepoint=5, n_dim=5): 
    """
    use to reshape prediction
    """
    nday = x.detach().cpu().numpy().reshape(n_timepoint,-1)

    if nday.shape[-1] != t5_ad.shape[0]:
        nday = nday.reshape(n_timepoint, -1, n_dim)
    return nday



def compute_guassian_u(Cellstate_ay, dimension=10):
    r"""
    Estimating the density high dimensional cell state coordinates and its change by experimental time.
    The density estimation function is 
    
    Input
    -------
    Cellstate_ay : ndarray of shape (N_cell, dimension), the high dimensional cell state representation, i.e. PC, Diffusion Map (DM) , SCVI-latent
    dimension : control the nubmer of the first few dimension to use for density esitmation

    Return
    -------
    gussian_kde_u : ndarray of shape (N_cell, 1), the density of each cell 

    Example
    ------
    >>> timepoints = sorted(ad.obs.timepoint_tx_days.unique()) # get timepoints
    >>> cbs_t = [ad.obs.query("`timepoint_tx_days` == @t").index for t in timepoints]
    >>> DM_ay = [ad[cb].obsm['DM_eigen'] for cb in cbs_t]
    >>> gussian_kde_u = guassian_u(DM_ay, dimension=5)
    >>> ad.obs['DM5_gaussisn_u']= gussian_kde_u

    """
    gussian_kde_u = []

    for m in Cellstate_ay:
        # dimension truncated matrix
        dm = m[:, :dimension].T  
        
        # take in [n_dim, n_sample]
        kde_fn = gaussian_kde(dm, bw_method='silverman')       
        gussian_kde_u.append(kde_fn(dm))

    gussian_kde_u = np.concatenate(gussian_kde_u, axis=0)

    return gussian_kde_u

def boundary_density_at(D, t_b, x, density_fn:callable=None):
    """
    Evaluate the cell density at time point `t_b` using the pre-defined density function
    Input
    -------
    D: the extracted data dictionary
    t_b : the boundary time point
    x : the grided cell state coordiante
    density_fn : the pre-defined density function
    
    Return
    -------
    u_t : smoothed cell density over s and adjusted by pop size
    """

    if density_fn is not None:
        raise NotImplementedError("the predefined func is not sett up")
        # u_t, Nt, n_exp = predefine_density(D, t_b, x, density_fn)

    elif len(x.shape) == 1:
        u_t, Nt, n_exp = evaluate_1d_density(D, t_b, x)

    elif len(x.shape) == 2:
        u_t, Nt, n_exp = evaluate_2d_mesh_density(D, t_b, x)

    else:
        # for higher dimensional data
        # use the data point itself but not sampling from the entire space
        # u_t, Nt, n_exp = evaluate_2d_mesh_density(D, t_b, x)
        raise NotImplementedError("higher dimensional func is under development")
   
    return u_t, Nt, n_exp


def evaluate_1d_density(D, t_b, x):
    """
    extract the cell density at time point `t_b`, this function works for 1 dimensional data
    Input
    -------
    D: the extracted data dictionary
    t_b : the boundary time point
    x : the grided cell state coordiante
    
    Return
    -------
    u_t : smoothed cell density over s and adjusted by pop size
    """
    
    assert t_b in D['pop']['t'], "`t_b` is not the observed time point"
    
    # find the batch that belong to the tb time point
    tp_index = [i for i, t in enumerate(D['ind']['tp']) if t==t_b]  
    
    n_lib = len(tp_index) 
    
    n_grid = x.shape[0]
    
    ut = np.zeros((n_lib, n_grid))
    Nt = np.zeros(n_lib)
    
    for i0 in range(n_lib):
        
        i_hist = tp_index[i0]
        
        # compute the density function based on the pdt coord
        density = gaussian_kde(D['ind']['hist'][i_hist])
        
        # evaluate and normalize at grided time x
        ut[i0, :] = density(x)
        ut[i0, :] /= trapz(ut[i0, :], x)
        Nt[i0] = len(D['ind']['hist'][i_hist]) # n cells


    tb_index = np.where(D['pop']['t']==t_b)[0].item()
    u_t = np.mean(ut, axis=0) * D['pop']['mean'][tb_index]
    # u_t = 0.5 * (u_t[:-1] + u_t[1:])

    # n_exp = 1  #TODO: change n_exp ?
    n_exp = n_lib
    
    return u_t, Nt, n_exp


def evaluate_2d_mesh_density(D, t_b, x):
    """
    extract the cell density at time point `t_b`. this function works for 2 dimensional mesh grid 
    Input
    -------
    D: the extracted data dictionary
    t_b : the boundary time point
    x : the mesh grided cell state s1, s2
    
    Return
    -------
    u_t : smoothed cell density over s and adjusted by pop size
    """
    
    assert t_b in D['pop']['t'], "`t_b` is not the observed time point"
    
    # find the batch that belong to the tb time point
    tp_index = [i for i, t in enumerate(D['ind']['tp']) if t==t_b]  
    
    n_lib = len(tp_index) 
    
    n_grid = x.shape[0]  # for 2 d, i.e 300 * 300

    space = (x[1,0] - x[0,0])**2
    
    ut = np.zeros((n_lib, n_grid))
    Nt = np.zeros(n_lib)
    
    for i0 in range(n_lib):
        
        i_hist = tp_index[i0]
        
        # compute the density function based on the pdt coord
        # the input data is (#dim , #samples)
        density = gaussian_kde(D['ind']['hist'][i_hist].T)
        
        # evaluate and normalize at grided time x
        ut[i0, :] = density(x.T)
        ut[i0, :] /= trapz(ut[i0, :], dx=space)
        Nt[i0] = len(D['ind']['hist'][i_hist]) # n cells


    tb_index = np.where(D['pop']['t']==t_b)[0].item()
    u_t = np.mean(ut, axis=0) * D['pop']['mean'][tb_index]
    # u_t = 0.5 * (u_t[:-1] + u_t[1:])

    # n_exp = 1  #TODO: change n_exp ?
    n_exp = n_lib
    
    return u_t, Nt, n_exp

def compute_highdim_density(adata, cellstate_key='DM_EigenVectors', n_timepoints=None, n_dimension=None, timepoint_key='time', cellstate=None):
    
    timepoints = adata.uns['pop']['t'][:n_timepoints]
    n_timepoints = len(timepoints)

    if cellstate is None:
        cellstate = adata.obsm[cellstate_key][:, :n_dimension]
        
    popD = adata.uns['pop']

    u_ls = []
    density_funs = []

    for tb_idx, t_b in enumerate(popD['t']):
            
        # subset ad_t
        
        cb_t = adata.obs.query(f"`{timepoint_key}` == @t_b").index
        ad_t = adata[cb_t].copy()
        cellstate_t = ad_t.obsm[cellstate_key][:, :n_dimension]

        density_fun = gaussian_kde(cellstate_t.T)
        u  = density_fun(cellstate.T)   # evaluate with the entire space
        n_exp = popD['n_lib'][tb_idx]

        u_ls.append(u)
        density_funs.append(density_fun)

    return u_ls, density_funs

def evaluate_u_ds(cid, cellstate, u_tb, delta_s, den_fn, scaler):
    r"""
    func for multi-process density estimate inside time-point loop

    Input
    ------
    cid: cell index , numerical index, not cell barcode
    cellstate : array [n_cell, n_dim] high-dimensional coordinate (representation) of all cells
    u_tb : array [n_cell,] , the density of each cell at a timepoint
    delta_s : [n_dim, ] the small perturbation add to cell state
    den_fn : callable([n_dim, n_cell]) ,  the density estimation function
    scaler : density scaler , normlized 

    Return
    ------
    the duds : [n_cell, n_dim] , the density chagne along each dimension
    """
    cs = cellstate[cid]
    u_t = u_tb[cid]
    du_dcs_cell = []

    for i in range(cellstate.shape[1]):
        ds = np.zeros_like(cs)
        ds[i] = delta_s[i]
        s_prime = cs + ds
        u_prime = den_fn(s_prime.reshape(-1,1)) * scaler
        dudcs = (u_prime - u_t)/delta_s[i]
        du_dcs_cell.append(dudcs.reshape(1,1))

    return np.concatenate(du_dcs_cell, axis=1)

def augment_cdf(x, x_a, y):
    """
    This function takes a CDF defined on a grid x and augments it to a finer grid x_a by duplicating the CDF values within intervals defined by x. zeros or ones accordingly were added for where x_a values are outside the range of x
    """
    
    i_shift = 0
    y_a = y.copy()
    for i in range(len(x) - 1):
        i1 = len(np.where(x_a > x[i])[0]) + len(np.where(x_a < x[i + 1])[0]) - len(x_a)
        if i1 > 0:
            ones_matrix = np.ones((i1, y_a.shape[1]))
            
            # If any x_a within the interval, 
            # duplicates the CDF values at the current interval
            y_a = np.concatenate(
                (y_a[:i + i_shift + 1, :], y_a[i + i_shift, :] * ones_matrix, y_a[i + i_shift + 1:, :]), 
                axis=0)
            i_shift += i1
            
    
    # boundary condition :  x_a  are smaller than the minimum
    ia = np.where(x_a < x[0])[0]
    y_a = np.concatenate((np.zeros((len(ia), y_a.shape[1])), y_a), axis=0)
    
    # boundary condition :  x_a  are larger than the maximum
    ie = np.where(x_a > x[-1])[0]
    y_a = np.concatenate((y_a, np.ones((len(ie), y_a.shape[1]))), axis=0)
    return y_a


def Lambda1(epoch, gamma=0.2):
    """
    lr scheduler rule 1, quickly decay then steady
    """
    if epoch >= 5 :
        factor = np.exp((3-epoch**0.4))**gamma + 0.1
        
    else:
        factor = np.exp((5-epoch**0.6))

    return 3 if factor > 3 else factor 


def Lambda2(epoch, gamma=0.15, changepoint=150):
    """
    lr scheduler rule 2, decay then grow
    """

    factor_decay = np.exp((3-epoch**0.7))**gamma + 0.01

    changepoint_f = np.exp((3-changepoint**0.7))**gamma

    factor_grow = changepoint_f * np.exp((epoch/changepoint)**2)**gamma

    return factor_decay if epoch <= changepoint else factor_grow 

def traverse_neighbor(connectivities, k, knn_idx):
    k = -1*k 
    # random walk
    next_degree_knn = []
    for i_d in knn_idx:
        neighbor = np.argpartition(connectivities[i_d].A, k)[k:].tolist()
        next_degree_knn.extend(neighbor)
    
    # all neighbor visited to the current degree

    knn_idx_d_plus1 = knn_idx + next_degree_knn
    # knn_idx_d_plus1 = np.unique(knn_idx_d_plus1).astype(int)
    knn_idx_d_plus1 = np.unique(next_degree_knn).astype(int).tolist()
    return knn_idx_d_plus1

def _sample_by_distance(dist_array, candidate_idx, alpha=None, repeat=1):
    """
    based on the knn distance, sample the closest cell 
        P ~ (1 - distance)
    dist_array : ndarray, distance from all other cells to cell i
    candidate_idx : the index of knn / or cansidered neighbor cells
    alpha: parameter to adjust uncertainty, the lower the more uncertrain
    repeat: the number of cells to sample each time
    """
    alpha = 1 if alpha is None else alpha

    dist = dist_array[candidate_idx]
    where_inf = np.isinf(dist)
    if np.any(where_inf):
        try:
            next_max = dist[~where_inf].max()
            dist = np.where(where_inf, next_max, dist)
        except ValueError:
            dist = np.zeros_like(dist)

    knn_p = dist.max() - dist + 0.1*dist.min()
    if knn_p.sum() > 0:
        knn_p = knn_p**alpha
        p = knn_p / knn_p.sum() # normalized
    else:
        p = None 

    neighbor_idx = [np.random.choice(candidate_idx, p=p) for i in range(repeat)]
    if repeat == 1:
        neighbor_idx = neighbor_idx[0]
    return neighbor_idx, p

def sample_deltax(adata, max_degree=1, k=None, xkey=None, pseudotimekey='palantir_pseudotime', progressbar=True):
    """
    the Key function defines the noise sampling process 
    given the starting point i
    """

    connectivities = adata.obsp['connectivities'].copy()
    distance = adata.obsp['distances'].copy()
    pdt = adata.obs[pseudotimekey].values

    if k is None:
        k  = adata.uns['neighbors']['params']['n_neighbors']
    
    if xkey is None:
        X = adata.X
    elif xkey in adata.obsm_keys():
        X = adata.obsm[xkey].copy()
    elif xkey in adata.layers():
        X = adata.layers[xkey].copy()

    def prograss_(x, turn_on=progressbar):
        if turn_on:
            return tqdm(x)
        else:
            return x
    
    delta_X = []
    neighbor_ls = []
    
    for i in prograss_(range(X.shape[0]), progressbar):

        knn_idx = np.array([i])
        t_i = pdt[i]

        # init
        pass_1 = 0
        pass_2 = 0
        n_degree = 0

        final_index = []
        
        # while pass_1*pass_2==0 and n_degree < max_degree:
        # for d in range(max_degree):  
            # if n_degree > self.free_search_degree:   # only under this degree can we expand knn without any constraints
            #     knn_idx = pass_2_idx
        
            # knn_idx = traverse_neighbor(connectivities, k, knn_idx) 
        knn_idx = np.argpartition(connectivities[i].A, -1*k)[-1*k:].tolist()

        # 1 : neighbor with the same condition
        pass_1_idx = knn_idx
        if len(pass_1_idx) > 0:
            pass_1 = 1
        else:
            final_index = [i]
            # continue

        # 2 : neighbor with bigger pseudo-time
        knn_t = pdt[pass_1_idx]
        if np.any(knn_t > t_i):
            pass_2_idx = np.array(pass_1_idx)[knn_t > t_i]
            final_index = pass_2_idx
            pass_2 = 1

        else:
            pass_2_idx = pass_1_idx
            final_index = [i]
            # continue
            # n_degree += 1
                    
        # sampled by distance

        
        if len(final_index) == 1:
            neighbor_idx = i
        else:
            # neighbor_idx, p = _sample_by_distance(distance, final_index)
            knn_p = connectivities[i,final_index].A.flatten()
            p = knn_p / knn_p.sum() if knn_p.sum() != 0 else None

            neighbor_idx = np.random.choice(final_index, p=p)

        delta_X.append( X[neighbor_idx] - X[i] )
        neighbor_ls.append(neighbor_idx)

    return delta_X, neighbor_ls


def make_coord_adata(adata, cellstate_key, v = None):
    r"""
    construct adata based on cellstate coodinates from expression matrix based adata
    the new adata is mainly for visualizing v

    Arguments:
    -----------
    """
    # create new adata with DM coordinate as X
    new_ad = ad.AnnData(
        X = adata.obsm['DM_EigenVectors_multiscaled'],
        obs = adata.obs.copy(),
        var = pd.DataFrame(['DM_%d'%d for d in range(5)]).set_index(0)
    )

    # transfer other highdim matrix
    new_ad.obsp['connectivities'] = adata.obsp['connectivities'].copy()
    new_ad.obsp['distances'] = adata.obsp['distances'].copy()
    new_ad.layers['cellstate'] = new_ad.X.copy()
    new_ad.obsm["X_pca"] = adata.obsm["X_pca"]
    new_ad.obsm["X_umap"] = adata.obsm["X_umap"]
    
    # pop info
    new_ad.uns = adata.uns.copy()
    timepoints = adata.uns['pop']['t']
    n_timepionts = len(timepoints)

    if v is not None:
        if v.shape[0] == new_ad.shape[0]:
            # single timepoint
            new_ad.layers['v'] = v
        elif v.shape[0] == n_timepionts: 
            assert len(v.shape) == 3, "please put in the raw v"
            for i, t in enumerate(timepoints):
                new_ad.layers[f'Day{t} v'] = v[i]

    return new_ad