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


class u_dt(pl.LightningModule):
    def __init__(self, lr, channels:list = [2, 32, 32, 1], activation_fn:Union[str, list] = 'Mish'):
        """
        mlp u theta
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = MLP_surrogate(channels=channels, activation_fn = activation_fn)
        self.lr = lr
        # self.loss_fn = nn.MSELoss(reduction='sum')

    def loss_fn(self, x, x_hat):
        r"""
        use L2 norm
        """
        if x.shape != x_hat.shape:
            x = x.squeeze()
            x_hat = x_hat.squeeze()
        assert x.shape == x_hat.shape

        loss = torch.norm(x-x_hat, p=2).mean()
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        return optimizer

    def forward(self, s, t):
        u = self.model(s, t) 
        u = torch.exp(u)
        return u
    
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

            dudt = torch.autograd.grad(u.sum(), t_in, create_graph=True)[0]

            # set ds to zeros to fix cellstates
            ds = torch.zeros_like(s).float().to(device).requires_grad_(True)

        return (dudt, ds)

    def training_step(self, train_batch, index):
        
        # cellstate, t, t+1, u_t, u_{t+1}
        s, (t, tp1), (ut, utp1) = train_batch

        # divided by 5 to reduce the integration time
        t0 = t[0].item() / 5
        t1 = tp1[0].item() / 5 

        device = s.device

        log_u_pred = self.model(s,t)

        # boundary u of the current timepoint
        ub_loss = self.loss_fn(ut.squeeze(), (torch.exp(log_u_pred).squeeze()))

        log_density_loss = self.loss_fn(torch.log(ut+1e-10), log_u_pred)

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

        total_loss = log_density_loss + log_utp1_loss

        with torch.no_grad():
            self.log_dict(
                {"boundary_loss":ub_loss.item(),
                 "log_boundary_loss": log_density_loss.item(),
                "integrat_loss":utp1_loss.item(),
                "log_integrat_loss":log_utp1_loss.item()
                })
            self.log("total_loss", total_loss.item(), prog_bar=True)

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


class u_dt_weight(u_dt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
