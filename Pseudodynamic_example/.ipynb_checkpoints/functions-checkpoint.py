import numpy as np
import torch

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


def h_poly(t):
    
    # zero order, first order, second order, third order
    tt = t[None, :]**torch.arange(4, device=t.device)[:, None]
    A = torch.tensor([
        [1, 0, -3, 2],
        [0, 1, -2, 1],
        [0, 0, 3, -2],
        [0, 0, -1, 1]
    ], dtype=t.dtype, device=t.device)
    return A @ tt

def spine_fun(hh, dx, y, idxs, m):
    """
    the main function doing the calculation
    ----------
    
    """
    cs = hh[0] * y[idxs] + hh[1] * m[idxs] * dx + hh[2] * y[idxs + 1] + hh[3] * m[idxs + 1] * dx
    
    return cs


def interp(x, y, xs):
    """
    
    """
    # 
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
    
    #
    idxs = torch.searchsorted(x[1:], xs)
    dx = (x[idxs + 1] - x[idxs])
    hh = h_poly((xs - x[idxs]) / dx)
    
    #
    interp_value = spine_fun(hh, dx, y, idxs, m)
    
    return interp_value