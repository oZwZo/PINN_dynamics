import os,sys
import numpy as np
import pandas as pd
import torch
from torch import nn
import pytorch_lightning as pl
from typing import Any, Union


class PINN_base(pl.LightningModule):
    def __init__(self, u:nn.Module , n_grid:int = 300, lr: Union[float, int] = 3e-4, optim_class="Adam"):
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
        
        # PDE discretization
        self.n_grid = 300
        grid = np.linspace(0, 1, n_grid)
        self.grid = grid
        h_inv = (1 / (grid[1] - grid[0]))
        
        # interval ∆s := s[i+1] - s[i]
        grid_s = np.linspace(grid[0] + (grid[1] - grid[0]) / 2, grid[-2] + (grid[-1] - grid[-2]) / 2, n_grid - 1)
        self.grid_s = torch.from_numpy(grid_s)
        
        # inverse interval
        self.register_buffer("h_inv", torch.tensor(h_inv, dtype=torch.float32, requires_grad=False))
        self.register_buffer("h2inv", torch.tensor(h_inv**2, dtype=torch.float32, requires_grad=False))
        
        
        # optimization and loss
        self.lr = lr
        self.optim_class = optim_class
        self.PopL_fn = nn.GaussianNLLLoss()                     # for population loss
        self.SSE_fn = nn.MSELoss(reduction='sum')               # for residual and boundary loss
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
        
        return optimizer
    
    def fowrard(self, s, t) -> torch.Tensor:
        """
        use the neural network to evaluate the density 
        """
        return self.u(s,t)
    
    def formular(self, s, t) -> tuple:
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
        lhs = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        
        
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
    
    def simplified_formular(self, s, t) -> tuple:
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
        lhs = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        
        # the first order deviritives of density u to cell state : ∂u/∂s
        duds = torch.autograd.grad(u.sum(), s, create_graph=True)[0]
        
        # the second order deviritives of density u to cell state : ∂^2u/∂s^2
        u_ss = torch.autograd.grad(duds.sum(), s, create_graph=True)[0]
        
        # right hand side
        rhs = torch.mul(D, u_ss) + torch.mul(v, duds) + torch.mul(g, u)
        
        return lhs, rhs

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
        
        # the probability of prediction : non-negative
        u_pred_b = u_pred_b - u_pred_b.min(axis=1)[0].view(-1,1) + 1e-17
        p_pred_b = u_pred_b / u_pred_b.sum(axis=1,keepdim=True)
        
        # prediction should be a distribution in the log space
        #.         y pred  ,  y_true
        L_kld = self.KLD_fn(p_pred_b.squeeze().log(), p_b.squeeze())
        
        return L_kld.mean(axis=1).sum()
    
    
    def boundary_loss(self, u_pred_b, u_b) -> torch.Tensor:
        """
        the loss defined at boundary conditions: including initial conditions, boundary conditions
        
        Input
        ------
        u_pred_b : u predicted at boundary timepoint
        u_b : observed boundary
        """
        return self.SSE_fn(u_pred_b, u_b) 
    
    def risidual_loss(self, s, t) -> torch.Tensor:
        """
        calculate the loss for collocation points, this loss inject the pde into the neural network
        
        Input
        ------
        s: the cell state, 
        t: experimental time
        """
        lhs, rhs = self.simplified_formular(s, t)
        return self.SSE_fn(rhs, lhs)
        
    def population_loss(self, u_pred, Mean, Var) -> torch.Tensor:
        """
        the loss term defined for population size, governed by Gaussian Negative Likelihood loss
            Gaussian NLL := 0.5 * log(var) + 0.5 * (input−target)**2/var  +const
        
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
        assert u_pred.shape[1] == self.n_grid , "make sure the same grid is applied"
        
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

        t_b = torch.einsum('ijk->jki', t_b).float()      # change dimension
        s_all = torch.einsum('ijk->jki', s_all).float()  # (1, T, n_grid) -> (T, n_grid, 1)

        # reguires_grad
        if requires_grad:
            s_col.requires_grad = True
            t_col.requires_grad = True
            s_all.requires_grad = True
            t_b.requires_grad = True

        Mean = Mean.T.float()
        Var = Var.T.float()

        return s_col, t_col, s_all, t_b, u_b, Mean, Var
    
    def training_step(self, train_batch, index):
        """
        get the data and compute the loss
        """
        s_col, t_col, s_all, t_b, u_b, Mean, Var = self.get_data(train_batch)
        
        # predict at boundary time poits
        u_pred_b = self.u(s_all, t_b)
        
        Loss_b = self.boundary_loss(u_pred_b, u_b)
        Loss_k = self.distribution_loss(u_pred_b, u_b)
        Loss_p = self.population_loss(u_pred_b, Mean, Var)
        
        
        # residual loss defied on collocation points
        Loss_r = self.risidual_loss(s_col, t_col)
        
        Loss_total = Loss_r + Loss_k + Loss_b + Loss_p
        
        self.log("residual_loss", Loss_r, on_epoch=True)
        self.log("distribution_loss", Loss_k, on_epoch=True)
        self.log("boundary_loss", Loss_b, on_epoch=True)
        self.log("population_loss", Loss_p, on_epoch=True)
        self.log("total_loss", Loss_total, on_epoch=True)
        
        return Loss_total
        
        
    
    def validation_step(self, val_batch, index):
        
        s_col, t_col, s_all, t_b, u_b, Mean, Var = self.get_data(val_batch, requires_grad=False)
    
        # predict at boundary time poits
        u_pred_b = self.u(s_all, t_b)
        
        Loss_b = self.boundary_loss(u_pred_b, u_b.squeeze())
        Loss_k = self.distribution_loss(u_pred_b, u_b)
        Loss_p = self.population_loss(u_pred_b, Mean, Var)
    
        # residual loss defied on collocation points
        # Loss_r = self.risidual_loss(s_col, t_col)
        Loss_r = 20
        
        Loss_total = Loss_k + Loss_b + Loss_p
        
        self.log("residual_loss", Loss_r, on_epoch=True)
        self.log("distribution_loss", Loss_k, on_epoch=True)
        self.log("boundary_loss", Loss_b, on_epoch=True)
        self.log("population_loss", Loss_p, on_epoch=True)
        self.log("total_loss", Loss_total, on_epoch=True)

        return Loss_total
        
    
    def test_step(self, test_databatch, index):
        
        s_col, t_col, s_all, t_b, u_b, Mean, Var = self.get_data(test_databatch)
        
        
        # predict at boundary time poits
        u_pred_b = self.u(s_all, t_b)
        
        Loss_b = self.boundary_loss(u_pred_b, u_b)
        Loss_k = self.distribution_loss(u_pred_b, u_b)
        Loss_p = self.population_loss(u_pred_b, Mean, Var)
        
        
        # residual loss defied on collocation points
        Loss_r = self.risidual_loss(s_col, t_col)
        
        Loss_total = Loss_r + Loss_k + Loss_b + Loss_p
        
        self.log("residual_loss", Loss_r)
        self.log("distribution_loss", Loss_k, on_epoch=True)
        self.log("boundary_loss", Loss_b, on_epoch=True)
        self.log("population_loss", Loss_p, on_epoch=True)
        self.log("total_loss", Loss_total, on_epoch=True)
        
        return Loss_total