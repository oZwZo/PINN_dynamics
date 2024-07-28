import os,sys
import numpy as np
import pandas as pd
import torch
from torch import nn
import pytorch_lightning as pl
from typing import Any, Union
from ._PINN_base import PINN_base

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
        
        # a lot of sanity check
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, requires_grad=True)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, requires_grad=True)

        #  check input shape
        if len(t.shape) == len(s.shape)-1: 
            # t is just flatten but s is high dimensional
            t = t.unsqueeze(-1)

        if type(t) == int:
            t = torch.full_like(s, fill_value=t, device=s.device, requires_grad=s.requires_grad)
        # if t.shape[-1] != 1:
        #     t = t.unsqueeze(-1)

        assert len(s.shape) == len(t.shape), "make sure s and t has the same shape"
        input = torch.cat([s,t], dim=-1)

        out = self.u_theta(input)
        return out.squeeze(-1)  # -> (B, n_grid)

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
    def __init__(self, y, x=None, n_knot=None):
        """
        High dimensional Cubic Spline with independent knots per dimension
        x : the coordinate space
        y : 2-d array/tensor values of the initial knots
        """
        super().__init__()
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
            self.Splines.append(CubicSpline(y=y[:,d2], x=x, n_knot=y.shape[0]))

        # self.Splines = nn.ModuleList(Splines)
        
    
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
        return torch.cat(ys, axis=1)

class MLP(pl.LightningModule):
    """
    MLP surrogate wrap by Lightning Module    
    """
    def __init__(self, lr, channels:list = [2, 32, 32, 1], activation_fn:Union[str, list] = 'Mish'):
        super().__init__()
        self.save_hyperparameters()
        self.model = MLP_surrogate(channels=channels, activation_fn = activation_fn)
        self.lr = lr
        self.loss_fn = nn.MSELoss(reduction='sum')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def forward(self, s, t):
        return self.model(s, t)
    
    def training_step(self, train_batch, index):
        s,t, u = train_batch
        u_pred = self.model(s,t)
        
        Total_loss = self.loss_fn(u.squeeze(), u_pred.squeeze())

        self.log("total_loss", Total_loss, on_epoch=True, prog_bar=True)
        return Total_loss



class MLP_exp(MLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, s, t):
        base = self.model(s, t)
        return torch.exp(base)

class Cspline_PINN(PINN_base):
    def __init__(self, *, n_knot=9,  n_dim=1, **kwargs):
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
                
            self.D = MultiDim_CubicSpline(y = torch.ones((n_knot, n_dim)).float(), n_knot=n_knot)
            self.v = MultiDim_CubicSpline(y = vy, n_knot=n_knot)
            self.g = MultiDim_CubicSpline(y = torch.ones((n_knot, n_dim)).float(), n_knot=n_knot)
        
    def get_data(self, data_batch, requires_grad=True):
        s_col, t_col, s_bon, t_bon, u_bon = data_batch

        s_col = s_col.squeeze().float()
        t_col = t_col.squeeze().float()

        t_bon = t_bon.squeeze(dim=0).float() if len(t_bon.shape) == 4 else t_bon  # change dimension

        s_bon = s_bon.squeeze(dim=0).float() if len(s_bon.shape) == 4 else s_bon # torch.einsum('ijk->jik', s_bon).float()
        # if cell state has higher dimension
        # (1, T, n_grid) -> (T, n_grid, 1)

        # reguires_grad
        if requires_grad:
            s_col.requires_grad = True
            t_col.requires_grad = True
            s_bon.requires_grad = True
            t_bon.requires_grad = True

        return s_col, t_col, s_bon, t_bon, u_bon

    def compute_loss(self, batch_data):
        """
        get the data and compute the loss

        Return
        -------
        residual loss
        boundary loss
        population loss
        """

        Loss_p = 0
        Loss_k = 0

        s_col, t_col, s_bon, t_bon, u_bon = self.get_data(batch_data)
        
        # predict at boundary time poits
        u_pred_b = self.u(s_bon, t_bon)
        Loss_b = self.boundary_loss(u_pred_b, u_bon)

        # residual loss defied on collocation points
        Loss_r = self.risidual_loss(s_col, t_col)
        
        return Loss_r, Loss_b, Loss_p, Loss_k


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

    def training_step(self, train_batch, index):
        """
        log individual loss term and them combine then into total loss
        """
        Loss_r, Loss_b, Loss_p, Loss_k = self.compute_loss(train_batch)
        
        Loss_total = Loss_b +  Loss_r  # only two loss is used here
        
        
        # if Loss_total < 1e-6:
        #     Loss_total *= 1000
        # if Loss_total < 1e-5:
        #     Loss_total *= 100
        # elif Loss_total < 1e-4:
        #     Loss_total *= 10
        # elif Loss_total < 1e-3:
        #     Loss_total *= 2
        
        
        self.log("residual_loss", Loss_r, on_epoch=True)
        self.log("boundary_loss", Loss_b, on_epoch=True)
        self.log("population_loss", Loss_p, on_epoch=True)
        self.log("total_loss", Loss_total, on_epoch=True)

        if self.schedule_lr != "False":
            self.log("lr",self.scheduler.get_last_lr()[0], on_epoch=True)
        
        return Loss_total


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

    def training_step(self, train_batch, index):
        """
        log individual loss term and them combine then into total loss
        """
        Loss_r, Loss_b, Loss_p, Loss_k = self.compute_loss(train_batch)
        
        Loss_total = Loss_b 
        
        
        self.log("residual_loss", Loss_r, on_epoch=True)
        self.log("boundary_loss", Loss_b, on_epoch=True)
        self.log("population_loss", Loss_p, on_epoch=True)
        self.log("total_loss", Loss_total, on_epoch=True)

        if self.schedule_lr != "False":
            self.log("lr",self.scheduler.get_last_lr()[0], on_epoch=True)
        
        return Loss_total

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
    