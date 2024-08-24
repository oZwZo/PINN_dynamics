import os,sys
import numpy as np
import pandas as pd
import torch
from torch import nn
import pytorch_lightning as pl
from typing import Any, Union
from ._PINN_base import PINN_base, PINN_base_sim
from .MLP_models import MLP_surrogate
from typing import Any, Union, Callable
from torchdiffeq import odeint


class pde_params(pl.LightningModule):
    def __init__(self, lr, channels, collapse_D = True, collapse_v = False, g_channels=None, v_channels=None, D_channels=None, time_sensitive=True, activation_fn:Union[str, list] = 'Mish', D_penalty = None):
        """
        mlp u theta

        Arguments:
        -------------
        channel : the number of MLP channels of the Behavior function
        [g, v, D]_channel : the number of MLP channels of the Behavior function
        collapse_[D,v] : merge the multi-channel output into 1 channel, 
                         which controls the complexity of the pde term.
        
        kwargs 
        -------
        u_theta : the neural netowrk surrogate of u
        lr: float, the learning rate
        optim_class : str, the optimizer used
        D_penalty : float , default None the weight for penalizing D


        """
        super().__init__()
        self.save_hyperparameters()
        
        self.time_sensitive = time_sensitive
        self.D_penalty = 0.1 if D_penalty is None else D_penalty

        if time_sensitive:
            self.n_dim = channels[0] - 1  # the fist dimension is (s, t)
            MLP_Module = MLP_surrogate
        

        # the output for growth is always 1
        if g_channels  is None:
            g_channels = channels + [1]
        self.g = MLP_Module(channels = g_channels, activation_fn='Tanh')

        # if we choose to collapse v, that means the parameter is the same for all dimension
        if v_channels is None:
            v_channels = channels + [1] if collapse_v else channels + [self.n_dim]
        self.v = MLP_Module(channels = v_channels, activation_fn='Tanh')

        # if we choose to collapse D, that means the parameter is the same for all dimension
        if D_channels is None:
            D_channels = channels + [1] if collapse_D else channels + [self.n_dim]
        self.D = MLP_Module(channels = D_channels, activation_fn='Tanh')

        

    def loss_fn(self,x, x_hat, weight=None):
        """
        both x and x_hat are log transformed
        """
        # sanity check
        if x.shape != x_hat.shape:
            x = x.squeeze()
            x_hat = x_hat.squeeze()
        assert x.shape == x_hat.shape
 
        if torch.all(x >0) :
            x  = torch.log(x + 1e-9)
        if torch.all(x_hat > 0):
            x_hat  = torch.log(x_hat + 1e-9)
        
        # -24 is ~ log(1e-9)
        x = torch.clamp(x, min=-24) 
        x_hat = torch.clamp(x_hat, min=-24)

        if weight == None:
            weight = (23+x)**3
            weight /= weight.sum()

        # compute loss
        loss = torch.sum(weight * (x - x_hat) ** 2)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam([module.parameters() for module in  [self.g, self.v, self.D]], lr=self.lr)
        return optimizer

    def forward(self, s, t):
        u = self.model(s, t) 
        u = torch.exp(u)
        return u

    def trace_div(self, f, s):
        """
        Calculates the Divergence : which is the trace of the Jacobian df/ds.
        f :  f(s), the output of a function
        s :  s, the variable on which to calculating the derivitives

        Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
        """
        sum_diag = 0.
        for i in range(s.shape[1]):
            sum_diag += torch.autograd.grad(f[:, i].sum(), s, create_graph=True)[0].contiguous()[:, i].contiguous()

        return sum_diag.contiguous()

    def mul(self, param, term):
        """
        own multiply function to deal with different dimension
        """

        # (bs, )  or (bs, n_dim)
        if param.shape == term.shape:
            prod = torch.mul(param, term)
    
        elif len(term.shape) == 1:
            prod = torch.mul(param, term.unsqueeze(1))
        
        elif len(param.shape) == 1:
            prod = torch.mul(param.unsqueeze(1), term)

        if len(prod.shape) != 1:
            prod = prod.sum(dim=1)

        return prod
        

    def equation(self, s, t) -> tuple:
        """
        Apply torch's auto grad to compute the dynamics
        
        based on the following equation:
            ∂u/∂t = ∂/∂s[ D* ∂u/∂s ] - ∂/∂s[ v*u ] + g*u
        
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
        Du = self.mul(D, duds)   # element-wise 
        
        # right hand side
        if len(Du.shape) == 1: # for one trajectory system
            # the second order deviritives of density u to cell state : ∂^2u/∂s^2
            #  ∂/∂s (D*∂u/∂s)
            d2Dds2 = torch.autograd.grad(Du.sum(), s, create_graph=True)[0] 

        else:   # for multi-dimensiona data
            # u_ss is different for multi dimension : ∂2u / ∂s_is_i 
            d2Dds2_ls  = []
            for i in range(v.shape[1]):
                du_dsisi = torch.autograd.grad(Du[:,i].sum(), s, create_graph=True)[0][:, i:i+1]
                d2Dds2_ls.append(du_dsisi)
            d2Dds2 = torch.cat(d2Dds2_ls, dim=1)
        
        # the second term : ∂/∂s[ v*u ]
        vu = self.mul(v, u)
        dvuds = torch.autograd.grad(vu.sum(), s, create_graph=True)[0] #TODO:check shape
        
        # right hand side
        diffuse = d2Dds2.sum(dim=1)
        drift = dvuds.sum(dim=1)
        growth = torch.mul(g, u)
        
        return dudt, growth, drift, diffuse
    
    
    def ode_func(self, t, states):
        """
        the function used for odeint
        """
        s = states[1]
        device = s.device
        t_in = torch.full((s.shape[0],1), t.item()*5).float().to(device)

        with torch.set_grad_enabled(True):

            s.requires_grad_(True)
            t_in.requires_grad_(True)

            u = self.forward(s, t_in) # make sure it is u but not log u

            _, growth, drift, diffuse = self.equation(s, t_in)

            dudt = growth - drift + diffuse

            # set ds to zeros to fix cellstates
            ds = torch.zeros_like(s).float().to(device).requires_grad_(True)

        return (dudt, ds)

    def restrict_D(self, s, t):
        """
        penalize D to restrict instability
        """
        D = self.D(s,t)
        D_L2 = torch.norm(D, p=2).sum()  # in case D is high dimensional
        return D_L2 


    def training_step(self, train_batch, index):
        
        # cellstate, t, t+1, u_t, u_{t+1}
        s, (t, tp1), (ut, utp1) = train_batch

        # divided by 5 to reduce the integration time
        t0 = t[0].item() / 5
        t1 = tp1[0].item() / 5 

        device = s.device


        # loss 1 : boundary loss
        log_u_pred = self.model(s,t)
        log_utp1_pred = self.model(s,tp1)

        # boundary u of the current timepoint
        ub_loss = self.loss_fn(ut.squeeze(), (torch.exp(log_u_pred).squeeze()))
        log_density_loss_t = self.loss_fn(torch.log(ut+1e-10), log_u_pred)
        log_density_loss_tp1 = self.loss_fn(torch.log(utp1+1e-10), log_utp1_pred)

        
        # loss 2 : dynamics 

        # init_condition 
        init_condition = (ut, s)
        step_size = np.around((t1 - t0)/15, decimals=1).item() 
        step_size = step_size if step_size > 0 else 0.05
        step_size = min(step_size, 0.4)

        u_int, s_t = odeint(
                        self.ode_func,
                        init_condition,
                        torch.tensor([t0, t1]).type(torch.float32).to(device),
                        atol=1e-5,
                        rtol=1e-5,
                        method='midpoint',
                        options = {'step_size': step_size}
                    )

        # boundary u of  the next timepoint
        utp1_loss = self.loss_fn(u_int[-1], utp1)
        u_int = nn.functional.relu(u_int)

        log_utp1_loss = self.loss_fn(torch.log(utp1+1e-10), torch.log(u_int[-1]+1e-10))

        D_norm = self.restrict_D(s,t)

        total_loss = log_density_loss_t + log_density_loss_tp1 + 2 * log_utp1_loss + self.D_penalty * D_norm


        with torch.no_grad():
            # self.log("residual_loss", Loss_r, on_epoch=True)
            # self.log("boundary_loss", Loss_b, on_epoch=True)
            # self.log("population_loss", Loss_p, on_epoch=True)
            
            self.log("boundary_loss", ub_loss.item(),  on_epoch=True)
            self.log("log_boundary_loss",  log_density_loss_t.item(),  on_epoch=True)
            self.log("integrat_loss", utp1_loss.item(),  on_epoch=True)
            self.log("log_integrat_loss", log_utp1_loss.item(), on_epoch=True)
            self.log("total_loss", total_loss, on_epoch=True, prog_bar=True)

        return total_loss


    def validation_step(self, val_batch, index):

        # cellstate, t, t+1, u_t, u_{t+1}
        s, t, tp1, ut, utp1 = val_batch
        device = s.device

        u_pred = self.model(s,t)

        # boundary u of the current timepoint
        ub_loss = self.loss_fn(ut.squeeze(), u_pred.squeeze())

        init_condition = (ut, s)
        u_int, s_t = odeint(
                        self.ode_func,
                        init_condition,
                        torch.tensor([t, tp1]).type(torch.float32).to(device),
                        atol=1e-5,
                        rtol=1e-5,
                        method='midpoint',
                        options = {'step_size': 0.1}
                    )

        # boundary u of  the next timepoint
        utp1_loss = self.loss_fn(u_int[-1], utp1)
        
        total_loss = ub_loss + utp1_loss
        
        self.log_dict(
            {"boundary_loss":ub_loss,
             "integrat_loss":utp1_loss,
             "total_loss":total_loss})