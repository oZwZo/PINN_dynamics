import os,sys
import numpy as np
import pandas as pd
import torch
from torch import nn
import pytorch_lightning as pl
from typing import Any, Union
from PINN_base import PINN_base
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline

class MLP_surrogate(nn.Module):
    
    def __init__(self, channels:list = [2, 32, 32, 1], activation_fn:Union[str, list] = 'Mish'):

        super().__init__()

        ### activation function check
        if type(activation_fn) == str:
            assert activation_fn in dir(nn), "invalid activation function, please check `https://pytorch.org/docs/stable/nn.html`"
            self.act_fns = [activation_fn] * (len(channels)-1)

        elif type(activation_fn) == str:
            assert len(activation_fn) == len(channels) - 2 , "The length of activation_fn should be 2 less than channels"
            self.act_fns = activation_fn

        else:
            raise TypeError("Augment `activation_fn` can only be string or list")
        
        
        ### define MLP module
        self.u_theta = nn.Sequential()
        in_out = zip(channels[:-2], channels[1:-1])
        for i, channel in enumerate(in_out):
            self.u_theta.add_module(f"Linear_{i}", nn.Linear(*channel))
            self.u_theta.add_module(f"{self.act_fns[i]}_{i}", eval('nn.%s()'%activation_fn))

        self.u_theta.add_module(f"Linear_{i+1}", nn.Linear(*channels[-2:]))  # output layer
    
    def forward(self, s, t) -> torch.Tensor:
        
        #  check input shape
        if s.shape[-1] != 1:
            s = s.unsqueeze(-1)

        if type(t) == int:
            t = torch.full_like(s, fill_value=t, device=s.device, requires_grad=s.requires_grad)
        if t.shape[-1] != 1:
            t = t.unsqueeze(-1)

        assert s.shape == t.shape, "make sure s and t has the same shape"
        input = torch.cat([s,t], dim=-1)

        out = self.u_theta(input)
        return out.squeeze(-1)  # -> (B, n_grid)

class CubicSpline(nn.Module):
    def __init__(self, x=None, y=None, n_knot=11):
        """
        cubic hermit spine function for modelling cell behavior value
        """
        super().__init__()
        
        if x is None:
            x = np.linspace(0,1,n_knot)
        
        if y is None:
            y = torch.from_numpy(np.random.randn(n_knot,)).float()
     
        self.register_buffer("x", torch.tensor(x, dtype=torch.float32, requires_grad=False))
        self.y = torch.nn.parameter.Parameter(y, requires_grad=True)

        # parameters.guess = [parD*ones(9,1);-2;-2;-2;-2;-2;-4;-4;-10;-12;parA*ones(9,1)]
        # parameters.min = [-10.3616*ones(1,9),-11.5129*ones(1,9),-6*ones(1,9)];
        # parameters.max = [0*ones(1,9),0*ones(1,9),5*ones(1,9)];
        
    
    def h_poly(self,t) -> torch.Tensor:
        """
        t : x - x_i / (x_{i+1} - x_i), the closest residule
        """

        t = t.squeeze()
        
        A = torch.tensor([
            [1, 0, -3, 2],
            [0, 1, -2, 1],
            [0, 0, 3, -2],
            [0, 0, -1, 1]
        ], dtype=t.dtype, device=t.device)
        
        # zero order, first order, second order, third order
        if len(t.shape) == 1:
            tt = t[None, :]**torch.arange(4, device=t.device)[:, None]
            hh = A @ tt
        elif len(t.shape) == 2:
            tt = t[:, None, :]**torch.arange(4, device=t.device)[:, None]
            hh = torch.einsum("ij, bjk -> bik", A, tt)
        else:
            raise ValueError()
        return hh


    def forward(self, xs, t) -> torch.Tensor:
        """
        interpolat the value by granular cell state xs
        -------------
        xs: x inside small interval, cell state in our case
        t : real time, but used in CubicSpine interpolate
        """
        
        
        # slope 
        m = (self.y[1:] - self.y[:-1]) / (self.x[1:] - self.x[:-1])
        m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])

        # assign segment
        idxs = torch.searchsorted(self.x[1:], xs)
        dx = (self.x[idxs + 1] - self.x[idxs])
        hh = self.h_poly((xs - self.x[idxs]) / dx)

        # the main function doing the calculation
        cs = hh[0] * self.y[idxs] +\
             hh[1] * m[idxs] * dx +\
             hh[2] * self.y[idxs + 1] +\
             hh[3] * m[idxs + 1] * dx

        return cs
    
    
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
        
    
        
