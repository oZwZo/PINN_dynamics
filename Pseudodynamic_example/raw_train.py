import os
import numpy as np
import pandas as pd
import torch 
from torch import nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import callbacks 
from pytorch_lightning import loggers as pl_loggers

import PINNs
from reader import Pdyn_ExtractDataset





###               ###
#     read data     #
###               ###

pt_path = '/home/wergillius/Project/PINN_dynamics/Pseudodynamic_example/dataExample.pt'
train_DS = Pdyn_ExtractDataset(Data_pt=pt_path, n_grid=300, collocation_points=300, n_repeat=10, log_transform=False)

train_DL = DataLoader(train_DS, batch_size=1, num_workers=4)



###                  ###
#     define model     #
###                  ###

# define neural network surrogate
u_theta = PINNs.MLP_surrogate(channels = [2, 32, 32, 1], activation_fn='Tanh')

# pseudo dynamics model
Pdyn_model = PINNs.Cspline_PINN(u=u_theta, n_knot=11, lr=10)



###                     ###
#     define Triainer     #
###                     ###

device = 'gpu' if torch.cuda.is_available() else 'cpu'
gpu_device = 0

# without log transformation
pth_save_path = "/home/wergillius/Project/PINN_dynamics/logs/Cspline_woLogPop/"
tb_logger = pl_loggers.TensorBoardLogger(save_dir=pth_save_path)

trainer = pl.Trainer(auto_lr_find=True,
                     accelerator=device,
                     # fast_dev_run=True,
                     gradient_clip_val=1e3,
                     logger=tb_logger,
                     devices = [gpu_device],
                     max_epochs=1000,
                     callbacks=[callbacks.ModelCheckpoint(filename='{epoch}-{total_loss:.8f}',
                                                  monitor="total_loss", mode="min", save_top_k=2)]
                     )



Pdyn_model.train()
trainer.fit(Pdyn_model, train_DL)
        
####
##  give up using trainer
##  Training and Fitting
####

# iterations = 20000
# previous_validation_loss = 99999

# for epoch in range(iterations):
    
#     optimizer.zero_grad() # to maket the gradients zero
    
#     # Loss based on boundary conditions
#     pt_x_bc = Variable(torch.from_numpy(x_bc).float(), requires_grad=False).to(device)
#     pt_u_bc1 = Variable(torch.from_numpy(y_bc).float(), requires_grad=False).to(device)
#     pt_t_bc1 = Variable(torch.from_numpy(t_bc0).float(), requires_grad=False).to(device)

#     pt_t_bc0 = Variable(torch.from_numpy(t_bc0).float(), requires_grad=False).to(device)
#     pt_u_bc0 = Variable(torch.from_numpy(u_bc).float(), requires_grad=False).to(device)
    
#     net_bc_out0 = u_theta(pt_x_bc, pt_t_bc0) 
#     mse_b0 = mse_cost_function(net_bc_out0, pt_u_bc0)    # loss B

#     net_bc_out1 = u_theta(pt_x_bc, pt_t_bc1) 
#     mse_b1 = mse_cost_function(net_bc_out1, pt_u_bc1)    # loss B
    
#     # Loss based on PDE
#     x_collocation = np.random.uniform(low=0.0, high=2.0, size=(500,1))
#     t_collocation = np.random.uniform(low=0.0, high=1.0, size=(500,1))
#     all_zeros = np.zeros((500,1))
    
#     pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
#     pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
#     pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
    
#     f_out = f(pt_x_collocation, pt_t_collocation, u_theta) # output of collocation points
#     mse_f = mse_cost_function(f_out, pt_all_zeros)
    
#     # Combining the loss functions
#     loss = mse_b0 + mse_b1 + mse_f
    
#     loss.backward()
#     optimizer.step()
    
#     with torch.autograd.no_grad():
#         print(epoch, "Training Loss:", loss.data)