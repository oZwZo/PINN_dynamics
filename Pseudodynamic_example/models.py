import os,sys
import numpy as np
import pandas as pd
import torch
from torch import nn
import pytorch_lightning as pl
from typing import Any, Union

from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline


class CubicSpine(nn.Module):
    def __init__(self, x, y=None):
        """
        cubic hermit spine function for modelling cell behavior value
        """
        super().__init__()
        
        if y is None:
            y = torch.from_numpy(np.random.randb(11,))
     
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

    def spine_fun(self, hh, dx, idxs, m):
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

        return self.spine_fun(hh, dx, idxs, m)


class PINN_base(pl.LightningModule):
    def __init__(self, lr: Union[float, int] = 3e-4, optim_class="Adam"):
        """
        u_theta : the neural netowrk surrogate of u
        
        Arguments:
        lr: float, the learning rate
        optim_class : str, the optimizer used
        
        Return:
        the PINN
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        
        self.SSE_fn = nn.MSELoss(reduction='sum')
        
        # the neural netowrk surrogate of u
        self.u = nn.Sequential(
            nn.Linear(2, 16),
            nn.Sigmoid(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
        # behavior functions
#         self.v = ?
#         self.D = ?
#         self.g = ?
        
        
    
    def configure_optimizers(self):
        lr = self.lr
        if self.optim_class == 'LBFGS':
            optimizer = torch.optim.LBFGS(self.parameters(), lr=lr, max_iter=20,
                                          max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09)
        elif self.optim_class == 'RMSprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
        else:
            # i.e. Adam              
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    
    def fowrard(self, s, t):
        """
        use the neural network to evaluate the density 
        """
        return self.u(s,t)
    
    def formular(self, s, t):
        """
        Apply torch's auto grad to compute the 
        
        based on the following equation:
            ∂u/∂t = ∂/∂s[ D* ∂u/∂s ] + ∂/∂s[ v*u ] + g*u
        
        we calcuate the left hand side (lhs) and the right hand side
        """
        u = self.u(s,t)
        D = self.D(s,t)
        v = self.v(s,t)
        g = self.g(s,t)
        
        # left : ∂u/∂t
        dudt = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        
        
        # the first order deviritives of density u to time : ∂u/∂s
        duds = torch.autograd.grad(u.sum(), s, create_graph=True)[0]
        
        # the first term:  a second order derivative
        Du = torch.mul(D, duds)   # element-wise 
        d2Dds2 = torch.autograd.grad(Du.sum(), s, create_graph=True)[0] #TODO:check shape
        
        # the second term : ∂/∂s[ v*u ]
        vu = torch.mul(v, u)
        dvuds = torch.autograd.grad(vu.sum(), s, create_graph=True)[0] #TODO:check shape
        
        # right hand side
        rhs = d2Dds2 + dvuds + torch.mul(g, u)
        
        return lhs, rhs
    
    def simplified_formular(self, s, t):
        """
        Apply torch's auto grad to compute the 
        
        based on the following equation:
            ∂u/∂t = D * ∂^2u/∂s^2  + ∂u/∂s * v + g*u
        
        we calcuate the left hand side (lhs) and the right hand side
        """
        u = self.u(s,t)
        D = self.D(s,t)
        v = self.v(s,t)
        g = self.g(s,t)
        
        # left : ∂u/∂t
        dudt = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        
        # the first order deviritives of density u to cell state : ∂u/∂s
        duds = torch.autograd.grad(u.sum(), s, create_graph=True)[0]
        
        # the second order deviritives of density u to cell state : ∂^2u/∂s^2
        u_ss = torch.autograd.grad(duds.sum(), s, create_graph=True)[0]
        
        # right hand side
        rhs = torch.mul(D, u_ss) + torch.mul(v, duds) + torch.mul(g, u)
        
        return lhs, rhs
    
    
    def boundary_loss(self, s, t_b, u_b):
        """
        the loss defined at boundary conditions: including initial conditions, boundary conditions
        Input
        ------
        s: the cell state, 
        t_b: the boundary time point
        """
        # 1 : forward fit loss
        u_theta_b = self.u(s, t_b)
        
        # # 2 : discrete forward fit loss
        # if t_b == 0:
        #     L_discrete = 0
        # else:
        #     for t in np.arange(t_b-1, t_b, self.resolution):
        #         udt, _ = self.simplified_formular(s, t)
        
        L_boundary = self.SSE_fn(u_theta-u_b) 
        return L_boundary
    
    def risidual_loss(self, s, t):
        """
        calculate the loss for collocation points, this loss inject the pde into the neural network
        
        Input
        ------
        s: the cell state, 
        t: experimental time
        """
        lhs, rhs = self.simplified_formular(self, s, t)
        L_risidual = self.SSE_fn(rhs - lhs)
        return L_risidual
        
    def population_loss(self, s, t):
        L_pop = 0
        return L_pop
    
    
    
class Cspine_PINN(PINN_base):
    def __init__(self, lr: Union[float, int] = 3e-4, optim_class="Adam"):
        """
        The PINN that uses cubic spine to fit the behavior functions D(s,t), v(s,t) and g(s,t), while the u itself is still a neural network
        """
        
        super().__init__(lr=lr, optim_class=optim_class)
        
        self.D = CubicSpine(x)
        self.v = CubicSpine(x)
        self.g = CubicSpine(x)
        
