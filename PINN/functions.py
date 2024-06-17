import numpy as np
import torch
from scipy.stats import gaussian_kde,entropy
from scipy.integrate import trapz


def scale_dpt(dpt):
    """
    scale dpt array to 0 and 1
    """
    dpt_min = dpt.min()
    dpt_max = dpt.max()
    dpt_scaled = (dpt - dpt_min) / (dpt_max - dpt_min)
    
    return dpt_scaled

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
