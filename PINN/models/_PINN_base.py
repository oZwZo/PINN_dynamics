import os,sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import autograd
import pytorch_lightning as pl
from typing import Any, Union, Callable


class PINN_base(pl.LightningModule):
    def __init__(self, u:nn.Module , lr: Union[float, int] = 3e-4, optim_class="Adam", schedule_lr=False):
        """
        u_theta : the neural netowrk surrogate of u
        
        Arguments:
        lr: float, the learning rate
        optim_class : str, the optimizer used
        schedule_lr : str or callable, the way to control learnign rate scheduler
        
        Return:
        the PINN
        """
        super().__init__()
        self.save_hyperparameters()
        
        # PDE discretization
        self.schedule_lr = schedule_lr
        
        # optimization and loss
        self.lr = lr
        self.optim_class = optim_class
        self.PopL_fn = nn.GaussianNLLLoss()                     # for population loss
        self.L_norm_fn = nn.MSELoss(reduction='sum')               # for residual and boundary loss
        self.KLD_fn = torch.nn.KLDivLoss(reduction="none") # for distribution loss
        
        # the neural netowrk surrogate of u
        self.u = u
        
    
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
        
        if self.schedule_lr != "False":
            # so we can pass string
            if isinstance(self.schedule_lr, Callable):
                self.scheduler = self.schedule_lr(optimizer)

            else:
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size  = 10 , gamma = 0.1)
                
            return {
                "optimizer": optimizer,
                "lr_scheduler" : {
                    "scheduler" : self.scheduler,
                    "monitor" : "total_loss",
                    }
                }
        else:
            return optimizer
            
    
    def forward(self, s, t) -> torch.Tensor:
        """
        use the neural network to evaluate the density 
        """
        u = self.u(s,t)
        return u - u.min(axis=1)[0].view(-1,1)

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
    
    def simplified_equation(self, s, t) -> tuple:
        """
        Apply torch's auto grad to compute the 
        
        based on the following equation:
            ∂u/∂t = D * ∂^2u/∂s^2  + ∂u/∂s * v + g*u
        
        we calcuate the left hand side (lhs) and the right hand side
        """

        s.requires_grad = True
        t.requires_grad = True

        u = self.u(s,t)
        D = self.D(s,t)
        v = self.v(s,t)
        g = self.g(s,t) # d- dim

        u.requires_grad = True

        # v = nn.functional.relu(v)
        
        # left : ∂u/∂t
        dudt = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        
        # the first order deviritives of density u to cell state : ∂u/∂s
        duds = torch.autograd.grad(u.sum(), s, create_graph=True)[0]
        
        
        # right hand side
        if len(s.shape) == 1: # for one trajectory system
            
            # the second order deviritives of density u to cell state : ∂^2u/∂s^2
            #  ∂/∂s (D*∂u/∂s)
            u_ss = torch.autograd.grad(duds.sum(), s, create_graph=True)[0]

        else:   # for multi-dimensiona data

            # u_ss is different for multi dimension : ∂2u / ∂s_is_i 
            u_ss_ls  = []
            for i in range(v.shape[1]):
                du_dsisi = torch.autograd.grad(duds[:,i].sum(), s, create_graph=True)[0][:, i:i+1]
                u_ss_ls.append(du_dsisi)
            u_ss = torch.cat(u_ss_ls, dim=1)

            g = g if len(g.shape) == 1 else g.sum(dim=1)

            
        diffuse = self.mul(D, u_ss)
        drift = self.mul(v, duds)
        growth = self.mul(g, u)
                # self.trace_div(torch.mul(v, u.unsqueeze(-1)), s) +  \

        return dudt, growth, drift, diffuse

    def TIGON_equation(self, s, t) -> tuple:
        """
        Apply torch's auto grad to compute the 
        
        based on the following equation:
            ∂u/∂t = g * u - ∇ (v * u)
        
        we calcuate the left hand side (lhs) and the right hand side
        """
        u = self.u(s,t)
        # D = self.D(s,t)   # we wouldn't bother not having D
        v = self.v(s,t)
        g = self.g(s,t) # d- dim

        if len(g.shape) > 1:
            g = g.mean(dim=1)

        dudt = torch.autograd.grad(u.sum(), t, create_graph=True)[0]

        vu = v * u.reshape(-1, 1)   # (b, n_dim) * (b, 1)  -> (b, n_dim)

        diffuse = 0
        drift = self.trace_div(vu, s)

        growth = g * u

        return dudt, growth, drift, diffuse


    def log_TIGON_equation(self, s, t) -> tuple:
        """
        Apply torch's auto grad to compute the 
        
        based on the following equation:
            ∂log(u)/∂t = g - ∇ v 
        
        we calcuate the left hand side (lhs) and the right hand side
        """
        u = self.u(s,t)
        # D = self.D(s,t)   # we wouldn't bother not having D
        v = self.v(s,t)
        g = self.g(s,t) # d- dim

        u = torch.clamp(u, min=1e-10)

        if len(g.shape) > 1:
            g = g.mean(dim=1)

        dlnu_dt = torch.autograd.grad(torch.log(u).sum(), t, create_graph=True)[0]

        diffuse = 0
        drift = self.trace_div(v, s)

        growth = g 

        return dlnu_dt, growth, drift, diffuse

    # Area statistics
    def Area_loss(self, u_pred_b, u_b) -> torch.Tensor:
        # check `Pseudodynamic_example/llPseudodynamics.py:49`
        raise NotImplementedError

    def distribution_loss(self, u_pred_b, u_b) -> torch.Tensor:
        """
        the loss defined as the kl divergence of the distribution, used to keep the shape
        """
        # from density to probability
        u_b = u_b + 1e-17
        p_b = (u_b)/ u_b.sum(axis=1,keepdim=True)
        
        upred_dim = len(u_pred_b.shape)
        agg_axis = 0 if upred_dim == 1 else 1

        # the probability of prediction : non-negative
        u_pred_b = u_pred_b - u_pred_b.min(axis=agg_axis)[0].view(-1,1) + 1e-17
        p_pred_b = u_pred_b / u_pred_b.sum(axis=agg_axis,keepdim=True)
        
        # prediction should be a distribution in the log space
        #.         y pred  ,  y_true
        L_kld = self.KLD_fn(p_pred_b.squeeze().log(), p_b.squeeze())
        
        return L_kld.mean(axis=agg_axis).sum()
    
    
    def boundary_loss(self, u_pred_b, u_b) -> torch.Tensor: 
        """
        the loss defined at boundary conditions: including initial conditions, boundary conditions
        
        Input
        ------
        u_pred_b : u predicted at boundary timepoint
        u_b : observed boundary
        """
        return self.L_norm_fn(u_pred_b.squeeze(), u_b.squeeze()) 
    
    def risidual_loss(self, s, t) -> torch.Tensor:
        """
        calculate the loss for collocation points, this loss inject the pde into the neural network
        
        Input
        ------
        s: the cell state, 
        t: experimental time
        """
        dudt, growth, drift, diffuse = self.simplified_equation(s, t)
        rhs = growth + drift + diffuse
        return self.L_norm_fn(rhs.squeeze(), dudt.squeeze())
        
    def population_loss(self, u_pred, Mean, Var) -> torch.Tensor:
        """
        the loss term defined for population size, governed by Gaussian Negative Likelihood loss
            Gaussian NLL := 0.5 * log(var) + 0.5 * (input - target)**2/var  +const
        
        Arguments
        ---------
        u_pred : Tensor (t_obs, n_grid), the predicted density for all the cell states, self.u_theta(s_all, t_obs)
        Mean : Tensor (t_obs, 1), D['pop']['mean'], the mean of population size over repeat
        Var : Tensor (t_obs, 1), D['pop']['var'] / D['pop']['n_exp'] , the var of population size over repeat
        
        Return
        ---------
        L_pop : Tensor (1,), loss term summing all observed time point
        """
        
        # copying
        # assert u_pred.shape[1] == self.n_grid , "make sure the same grid is applied"
        
        # the estimated population size N_θ = ∫ u ds
        N_theta = 0.5*(u_pred[:,1:]+u_pred[:,:-1]).sum(dim=1, keepdim = True) #/ h_inv   
        # (t_obs, n_grid) -> (t_obs,1)
        
        # population 
        assert N_theta.shape == Mean.shape, 'input and target view not identical'
        
        # compute loss and sum for all observed time point
        L_pop = self.PopL_fn(input=N_theta, target=Mean, var=Var)

        return L_pop
    
    def get_data(self, data_batch, requires_grad=True):
        s_col, t_col, s_all, t_b, u_b, Mean, Var = data_batch

        s_col = s_col.squeeze().float()
        t_col = t_col.squeeze().float()

        t_b = t_b.squeeze(dim=0).float() if len(t_b.shape) == 4 else t_b  # change dimension

        s_all = s_all.squeeze(dim=0).float() if len(s_all.shape) == 4 else s_all # torch.einsum('ijk->jik', s_all).float()
        # if cell state has higher dimension
        # (1, T, n_grid) -> (T, n_grid, 1)

        # reguires_grad
        if requires_grad:
            s_col.requires_grad = True
            t_col.requires_grad = True
            s_all.requires_grad = True
            t_b.requires_grad = True

        Mean = Mean.squeeze(0).float() if Mean.shape[0] == 1 else Mean.squeeze(-1).float()
        Var = Var.T.float()

        return s_col, t_col, s_all, t_b, u_b, Mean, Var

    def compute_loss(self, batch_data):
        """
        get the data and compute the loss

        Return
        -------
        residual loss
        boundary loss
        population loss
        """
        s_col, t_col, s_all, t_b, u_b, Mean, Var = self.get_data(batch_data)
        
        # predict at boundary time poits
        u_pred_b = self.u(s_all, t_b)
        
        Loss_b = self.boundary_loss(u_pred_b, u_b)
        Loss_p = self.population_loss(u_pred_b, Mean, Var)
        Loss_k = self.distribution_loss(u_pred_b, u_b)
        # residual loss defied on collocation points
        Loss_r = self.risidual_loss(s_col, t_col)

        Loss_total = Loss_b + Loss_p + Loss_r 
        
        return Loss_total, Loss_r, Loss_b, Loss_p, Loss_k

    def training_step(self, train_batch, index):
        """
        log individual loss term and them combine then into total loss
        """
        Loss_total, Loss_r, Loss_b, Loss_p, Loss_k = self.compute_loss(train_batch)
    
        
        self.log("residual_loss", Loss_r, on_epoch=True)
        self.log("boundary_loss", Loss_b, on_epoch=True)
        self.log("population_loss", Loss_p, on_epoch=True)
        self.log("total_loss", Loss_total, on_epoch=True, prog_bar=True)

        if self.schedule_lr != "False":
            self.log("lr",self.scheduler.get_last_lr()[0], on_epoch=True)
        
        return Loss_total

    # def on_train_epoch_end(self):
    def validation_step(self, val_batch, index):
        
        Loss_total, Loss_r, Loss_b, Loss_p, Loss_k = self.compute_loss(val_batch)
    
        
        self.log("residual_loss", Loss_r, on_epoch=True)
        self.log("boundary_loss", Loss_b, on_epoch=True)
        self.log("population_loss", Loss_p, on_epoch=True)
        self.log("total_loss", Loss_total, on_epoch=True, prog_bar=True)

        if self.schedule_lr != "False":
            self.log("lr",self.scheduler.get_last_lr()[0], on_epoch=True)

        return Loss_total
    
    def predict_boundary(self,batch):
        """
        predicts density u and cell number N for the observed time points
        """

        s_col, t_col, s_all, t_b, u_b, Mean, Var = self.get_data(batch, False)


        # predict
        u_pred_b = self.u(s_all, t_b)

        u_pred_b = u_pred_b.detach().numpy()
        u_b = u_b.detach().numpy()[0]
        N_theta = 0.5*(u_pred_b[:,1:]+u_pred_b[:,:-1]).sum(axis=1)
        Mean = Mean.detach().numpy().flatten()
        Var = Var.detach().numpy().flatten()

        return u_pred_b, N_theta

    def predict_step(self, batch_data, batch_index):
        s_col, t_col, s_all, t_b, u_b = self.get_data(batch_data)
        # predict at boundary time poits
        u_pred_b = self.u(s_all, t_b)
        return  u_pred_b

    def predict_param(self, DataSet, param='g'):    
        """
        Given a DataSet Class, predict the param 
        """
        device = next(self.u.parameters()).device

        cellstate_only = (self.time_sensitive == False) and (param != 'u')
        
        # get cell states and their paired timepoints
        if cellstate_only:
            # cellstate is [n_cell, n_dim]
            s_all = torch.from_numpy(DataSet.cellstate).float().requires_grad_()
        else:
            # s is [n_time * n_cell , n_dim]
            s_all = DataSet.s.float().requires_grad_()
            

        t_b = DataSet.t_b.float().requires_grad_()

        # get the behavior function
        sub_module = self.__getattr__(param)
        param_pred = sub_module(s_all.to(device), t_b.to(device))

        if (len(param_pred.shape) != 1) and (param_pred.shape[1] == DataSet.n_dimension):
            param_pred = self.trace_div(param_pred, s_all)

            
        if cellstate_only:
            param_pred = param_pred.detach().cpu().numpy()
        else:
            n_timepoint = len(DataSet.popD['t'])
            param_pred = param_pred.detach().cpu().numpy().reshape(n_timepoint, -1)


        return param_pred

class PINN_base_sim(PINN_base):
    def __init__(self, u:nn.Module , lr: Union[float, int] = 3e-4, optim_class="Adam", schedule_lr=False):
        super().__init__(u=u, lr=lr, optim_class=optim_class, schedule_lr=schedule_lr)

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

    def forward(self, s, t) -> torch.Tensor:
        """
        use the neural network to evaluate the density 
        """
        u = self.u(s,t)
        return u #- u.min()[0].view(-1,1)

    def risidual_loss(self, s, t) -> torch.Tensor:
        """
        Diffusion is not used  
        
        Input
        ------
        s: the cell state, 
        t: experimental time
        """
        dudt, growth, drift, diffuse = self.simplified_equation(s, t)
        rhs = growth + drift 
        return self.L_norm_fn(rhs.squeeze(), dudt.squeeze())

    def compute_loss(self, batch_data):
        """
        get the data and compute the loss

        Return
        -------
        residual loss
        boundary loss
        population loss
        """
        s_col, t_col, s_all, t_b, u_b = self.get_data(batch_data)
        
        # predict at boundary time poits
        u_pred_b = self.u(s_all, t_b)
        
        Loss_b = self.boundary_loss(u_pred_b, u_b)
        Loss_p = 0
        Loss_k = 0
        # residual loss defied on collocation points
        Loss_r = self.risidual_loss(s_col, t_col)

        Loss_total = Loss_b + Loss_p + Loss_r 
        
        return Loss_total, Loss_r, Loss_b, Loss_p, Loss_k


def batch_jacobian(func, x, create_graph=False):
    """
    compute the jacobian matrix
    """
    def _func_sum(x):
        return func(x).sum(dim=0)
    return autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1, 0, 2)

def batch_hessian(func, x):
    """
    compute the hessian matrix
    """
    jacobian = batch_jacobian(func, x, create_graph=True)
    hessians = []
    for i in range(jacobian.size(1)):
        grad = autograd.grad(jacobian[:, i].sum(), x, create_graph=True, retain_graph=True)[0]
        hessians.append(grad.unsqueeze(1))
    return torch.cat(hessians, dim=1)



