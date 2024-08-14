import os,sys
import numpy as np
import pandas as pd
import torch
from torch import nn
import pytorch_lightning as pl
from typing import Any, Union
from ._PINN_base import PINN_base, PINN_base_sim

class MLP_surrogate(nn.Module):
    
    def __init__(self, channels:list = [2, 32, 32, 1], activation_fn:Union[str, list] = 'Mish'):

        super().__init__()
        self.time_sensitive = True

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

class MLP_s(MLP_surrogate):
    r"""
    MLP surrogate predicts merely with cellstate `s`
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_sensitive = False

    def forward(self, s, t):
        """
        `t` is not used but we keep args consistent
        """
        return self.u_theta(s).squeeze(-1)

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

class MLP_PINN(PINN_base_sim):
    def __init__(self, *, channels, collapse_D = True, collapse_v = False, g_channels=None, v_channels=None, D_channels=None, time_sensitive=True, **kwargs):
        r"""
        The PINN that uses MLP to fit all the functions D(s,t), v(s,t) and g(s,t), while the u itself is still a neural network
        
        Agument
        -------
        u : the u_theta , MLP_surrogate class
        channel : the number of MLP channels of the Behavior function
        [g, v, D]_channel : the number of MLP channels of the Behavior function
        collapse_[D,v] : merge the multi-channel output into 1 channel, 
                         which controls the complexity of the pde term.
        
        kwargs 
        -------
        u_theta : the neural netowrk surrogate of u
        lr: float, the learning rate
        optim_class : str, the optimizer used
        """
        super().__init__(**kwargs)
        self.time_sensitive = time_sensitive

        if time_sensitive:
            self.n_dim = channels[0] - 1  # the fist dimension is (s, t)
            MLP_Module = MLP_surrogate
        else:
            self.n_dim = channels[0]  
            MLP_Module = MLP_s
        

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


    def risidual_loss(self, s, t) -> torch.Tensor:
        """
        With MLP surrogate, the drift term is subsjected 
        calculate the loss for collocation points, this loss inject the pde into the neural network
        
        Input
        ------
        s: the cell state, 
        t: experimental time
        """
        dudt, growth, drift, diffuse = self.simplified_equation(s, t)
        rhs = growth - drift + diffuse
        return self.L_norm_fn(rhs.squeeze(), dudt.squeeze())


class MLP_woD(MLP_PINN):

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def risidual_loss(self, s, t) -> torch.Tensor:
        """
        Diffusion is not used  
        """
        dudt, growth, drift, diffuse = self.simplified_equation(s, t)
        rhs = growth - drift 
        return self.L_norm_fn(rhs.squeeze(), dudt.squeeze())

class MLP_logTIGON(MLP_PINN):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def risidual_loss(self, s, t) -> torch.Tensor:
        """
        Diffusion is not used  
        """
        dudt, growth, drift, diffuse = self.log_TIGON_equation(s, t)
        rhs = growth - drift 
        return self.L_norm_fn(rhs.squeeze(), dudt.squeeze())



class MLP_woD_logB(MLP_woD):

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def boundary_loss(self, u_pred_b, u_b) -> torch.Tensor: 
        """
        The boundary loss comes from two part
        
        Input
        ------
        u_pred_b : u predicted at boundary timepoint
        u_b : observed boundary
        """
        u_pred_b = torch.nn.functional.relu(u_pred_b).squeeze()
        u_b = u_b.squeeze()
        u_norm = self.L_norm_fn(u_pred_b, u_b) 


        # norm at log scale
        log_u_pred = torch.clamp(torch.log(u_pred_b+1e-30), min=-10, max=0)
        log_u_b = torch.clamp(torch.log(u_b), min=-10, max=0)
        log_u_norm = self.L_norm_fn(log_u_pred, log_u_b) 

        if self.current_epoch > 1:
            loss_b = u_norm + 0.001* log_u_norm
        else:
            loss_b = u_norm
        return loss_b
    
class MLP_woD_L2(MLP_woD):
    """
    use the L-infinity norm as the loss function
    """
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def L_norm_fn(self, a, b):
        return torch.norm(a.squeeze() - b.squeeze(), p=2)


class MLP_woD_Linf(MLP_woD):
    """
    use the L-infinity norm as the loss function
    """
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def L_norm_fn(self, a, b):
        return torch.norm(a.squeeze() - b.squeeze(), p=float('inf'))

class MLP_TIGON(MLP_PINN):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def risidual_loss(self, s, t) -> torch.Tensor:
        """
        Use the Tigon equation to inform the model
        """
        dudt, growth, drift, diffuse = self.TIGON_equation(s, t)
        rhs = growth - drift 
        return self.L_norm_fn(rhs.squeeze(), dudt.squeeze())
