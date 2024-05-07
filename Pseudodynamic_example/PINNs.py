import os,sys
import numpy as np
import pandas as pd
import torch
from torch import nn
import pytorch_lightning as pl
from typing import Any, Union
from PINN_base import PINN_base
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline


class CubicSpline(nn.Module):
    def __init__(self, x=None, y=None, n_knot=11):
        """
        cubic hermit spine function for modelling cell behavior value
        """
        super().__init__()
        
        if x is None:
            x = np.linespace(0,1,n_knot)
        
        if y is None:
            y = torch.from_numpy(np.random.randb(n_knot,))
     
        self.register_buffer("x", torch.tensor(x, dtype=torch.float32, requires_grad=False))
        self.y = torch.nn.parameter.Parameter(y, requires_grad=True)
        
    
    def h_poly(self,t):
        """
        t : x - x_i / (x_{i+1} - x_i), the closest residule
        """
        # zero order, first order, second order, third order
        tt = t[None, :]**torch.arange(4, device=t.device)[:, None]
        A = torch.tensor([
            [1, 0, -3, 2],
            [0, 1, -2, 1],
            [0, 0, 3, -2],
            [0, 0, -1, 1]
        ], dtype=t.dtype, device=t.device)
        return A @ tt

    
    def spline_fun(self, hh, dx, idxs, m):
        """
        the main function doing the calculation
        """
        y = self.y
        cs = hh[0] * y[idxs] + hh[1] * m[idxs] * dx + hh[2] * y[idxs + 1] + hh[3] * m[idxs + 1] * dx

        return cs


    def forward(self, xs, t):
        """
        interpolat the value by granular cell state xs
        -------------
        xs: x inside small interval, cell state in our case
        t : real time, but used in CubicSpine interpolate
        """
        x = self.x
        y = self.y
        
        # slope 
        m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
        m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])

        # assign segment
        idxs = torch.searchsorted(x[1:], xs)
        dx = (x[idxs + 1] - x[idxs])
        hh = self.h_poly((xs - x[idxs]) / dx)

        return self.spline_fun(hh, dx, idxs, m)
    
    
class Cspline_PINN(PINN_base):
    def __init__(self, u:nn.Module, n_knot=11, n_grid:int = 300, lr: Union[float, int] = 3e-4, optim_class="Adam"):
        """
        The PINN that uses cubic spine to fit the behavior functions D(s,t), v(s,t) and g(s,t), while the u itself is still a neural network
        
        Agument
        -------
        n_knot : the number of knots of the CubicSpline function
        
        """
        
        super().__init__(u=u, n_grid=n_grid, lr=lr, optim_class=optim_class)
        
        self.D = CubicSpline(n_knot=n_knot)
        self.v = CubicSpline(n_knot=n_knot)
        self.g = CubicSpline(n_knot=n_knot)
        
    
        
