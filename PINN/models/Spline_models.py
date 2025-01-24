import os,sys
import numpy as np
import pandas as pd
import torch
from torch import nn
import pytorch_lightning as pl
from typing import Any, Union
from ._PINN_base import PINN_base, PINN_base_sim

class CubicSpline(nn.Module):
    def __init__(self, x=None, y=None, n_knot=11):
        """
        cubic hermit spine function for modelling cell behavior value
        x : the coordinate space
        """
        super().__init__()
        
        if x is None:
            x = np.linspace(0,1,n_knot) #.reshape(n_knot,-1)
        # if len(x.shape) == 1:
        #     x = x.reshape(x.shape[0], -1)
        
        if y is None:
            y = torch.from_numpy(np.random.randn(n_knot,)).float()
     
        self.register_buffer("x", torch.tensor(x, dtype=torch.float32, requires_grad=False))
        self.y = torch.nn.parameter.Parameter(y, requires_grad=True)
        
    
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
        if not isinstance(xs, torch.Tensor):
            xs = torch.tensor(xs).float()
        
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

class MultiDim_CubicSpline(nn.Module):
    def __init__(self, y, x=None, n_knot=None, collapse=False):
        """
        High dimensional Cubic Spline with independent knots per dimension
        x : the coordinate space
        y : 2-d array/tensor values of the initial knots
        n_knot : the number of anchor points for each dimension. This will end up with
        collapse : bool, merge all the dimension into 1 output
        """
        super().__init__()

        self.collapse = collapse

        # sanity check
        if x is None:
            x = np.linspace(0,1, y.shape[0]) #.reshape(n_knot,-1)

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y).float()

        # sanity check
        assert len(y.shape) == 2,  "for MultiDim CubicSpline the initital y must be 2-d shape : [#knots, #dimensions]"
        assert y.shape[0] == x.shape[0] , "the number of anchor knots is not consistent between x and y"

        # define cublic spline for each column
        # Splines = []
        self.Splines = nn.ModuleList([])
        for d2 in range(y.shape[1]):
            self.Splines.append(CubicSpline(y=y[:,d2], x=x[:,d2], n_knot=y.shape[0]))

        if collapse:
            self.fc_out = nn.Linear(y.shape[1], 1, bias=False)
            weight = self.fc_out.weight
            n_in = self.fc_out.in_features
            self.fc_out.weight = nn.Parameter(torch.ones((1, n_in))).float()
        
    
    def forward(self, xs, t=None)->torch.Tensor:
        """
        Interpoloate for each axis of xs
        
        Arguments
        ------
        xs : [ndarray, tensor], high dimensional cell state spanning from 0 to 1 in each dimension

        Return:
        ys : tensor, interpolated y for each axis independently
        """

        assert len(xs.shape) == 2, 'Input xs must be 2-dim shape : [#points, #dimensions]'


        # forward the 1dim Cubic Spline for each dim
        ys = []
        for d2 in range(xs.shape[1]):
            ys_d2 = self.Splines[d2](xs[:, d2], t)      # out of the dimension
            ys.append(ys_d2.reshape(-1,1))              # make it 2dim for stacking
        out = torch.cat(ys, axis=1)

        if self.collapse:
            out = self.fc_out(out)
        return out


class Cspline_PINN(PINN_base_sim):
    def __init__(self, *, n_knot=9,  n_dim=1, D_collapse=False, v_collapse=False, **kwargs):
        """
        The PINN that uses cubic spine to fit the behavior functions D(s,t), v(s,t) and g(s,t), while the u itself is still a neural network
        
        Agument
        -------
        n_knot : the number of knots of the CubicSpline function
        
        kwargs 
        -------
        u_theta : the neural netowrk surrogate of u
        lr: float, the learning rate
        optim_class : str, the optimizer used
        """
        super().__init__(**kwargs)
        # super().__init__(u=u, n_grid=n_grid, lr=lr, optim_class=optim_class, schedule_lr=schedule_lr)
        self.n_dim = n_dim

        # 1 brach system
        if n_dim == 1:

            if n_knot == 9:
                vy = torch.from_numpy(np.array([-2,-2,-2,-2,-2,-4,-4,-10,-12])).float()
            else:
                vy = -2*torch.ones(n_knot)
                vy[-2] = -10
                vy[-1] = -12
                
            self.D = CubicSpline(y = torch.ones(n_knot).float(), n_knot=n_knot)
            self.v = CubicSpline(y = vy, n_knot=n_knot)
            self.g = CubicSpline(y = torch.ones(n_knot).float(), n_knot=n_knot)

        else:    # multi-brach system
            if n_knot == 9:
                v_init_base = [-2,-2,-2,-2,-2,-4,-4,-10,-12]
                vy = torch.from_numpy(np.array([v_init_base]*n_dim).T).float()
            else:
                vy = -2*torch.ones(n_knot)
                vy[-2] = -10
                vy[-1] = -12
                
            self.D = MultiDim_CubicSpline(y = torch.ones((n_knot, n_dim)).float(), n_knot=n_knot, collapse=D_collapse)
            self.v = MultiDim_CubicSpline(y = vy, n_knot=n_knot, collapse=v_collapse)
            self.g = MultiDim_CubicSpline(y = torch.ones((n_knot, n_dim)).float(), n_knot=n_knot, collapse=True)
    

class Cspline_woPL(Cspline_PINN):
    def __init__(self, *args, **kwargs):
        """
        The Cspline PINN model with out population loss
        
        Agument
        -------
        The same arguments as Cspline_PINN
        
        kwargs 
        -------
        u_theta : the neural netowrk surrogate of u
        lr: float, the learning rate
        optim_class : str, the optimizer used
        """
        super().__init__(*args, **kwargs)

    def compute_loss(self, batch_data):
        """
        re compute the total loss as only the boundary and residual loss
        """
        
        Loss_total, Loss_r, Loss_b, Loss_p, Loss_k = super().compute_loss(batch_data)

        Loss_total = 10*Loss_b + Loss_r
        
        return Loss_total, Loss_r, Loss_b, Loss_p, Loss_k


class Cspline_bo(Cspline_PINN):
    def __init__(self, *args, **kwargs):
        """
        The Cspline PINN model with boundary loss only
        
        Agument
        -------
        The same arguments as Cspline_PINN
        
        kwargs 
        -------
        u_theta : the neural netowrk surrogate of u
        lr: float, the learning rate
        optim_class : str, the optimizer used
        """
        super().__init__(*args, **kwargs)

    def compute_loss(self, batch_data):
        """
        re compute the total loss has the boundary loss onlu
        """
        
        Loss_total, Loss_r, Loss_b, Loss_p, Loss_k = super().compute_loss(batch_data)
        Loss_total = Loss_b
        
        return Loss_total, Loss_r, Loss_b, Loss_p, Loss_k

class Cspline_symKLD(Cspline_PINN):
    def __init__(self, u:nn.Module, n_knot=11, n_grid:int = 300, lr: Union[float, int] = 3e-4, optim_class="Adam", schedule_lr=None):
        """
        optimize the u_theta and cubic spline with an additional symmetric KLD loss

        Agument
        -------
        n_knot : the number of knots of the CubicSpline functio
        """
        super().__init__(u=u, n_knot=n_knot, n_grid=n_grid, lr = lr, optim_class=optim_class, schedule_lr=schedule_lr)
        

    def distribution_loss(self, u_pred_b, u_b) -> torch.Tensor:
        """
        the loss defined as the kl divergence of the distribution, used to keep the shape
        """
        # the density is non-negative
        u_b = u_b + 1e-17
        u_pred_b = u_pred_b - u_pred_b.min(axis=1)[0].view(-1,1) + 1e-17

        # KLD q-p, and KLD p-q
        L_kld = self.KLD_fn(u_pred_b.squeeze().log(), u_b.squeeze()).mean(axis=1).sum()
        L_kld2 = self.KLD_fn(u_b.squeeze().log(), u_pred_b.squeeze()).mean(axis=1).sum()
        
        return (L_kld + L_kld2) / 2
    

    def training_step(self, train_batch, index):
        """
        Add symmetric KLD loss to total loss
        """

        Loss_r, Loss_b, Loss_p, Loss_k = self.compute_loss(train_batch)


        if self.current_epoch >= 1:
            # only apply kld loss after the first epoch
            Loss_total = Loss_r + Loss_b + Loss_p + Loss_k
        else:
            Loss_total = Loss_r + Loss_b + Loss_p

        self.log("residual_loss", Loss_r, on_epoch=True)
        self.log("boundary_loss", Loss_b, on_epoch=True)
        self.log("population_loss", Loss_p, on_epoch=True)
        self.log("total_loss", Loss_total, on_epoch=True)
        
        return Loss_total
    