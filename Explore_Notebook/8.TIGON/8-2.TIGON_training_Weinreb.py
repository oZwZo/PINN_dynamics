import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import sys

# git clone TIGON 
sys.path.append('/ssd/users/Wergillius/Project/TIGON')
from utility import *

args = Args()
args.input_dir = "/ssd/users/Wergillius/Project/PINN_dynamics/data/TIGON_input"
args.gpu = '2'
args.dataset = 'klein_subset_DM_EigenVectors_multiscaled'
args.save_dir = '/ssd/users/Wergillius/Project/PINN_dynamics/logs/TIGON/model2'
args.niters = 500
args.lr =  3e-3
args.num_samples = 100
args.hidden_dim = 16
args.n_hiddens = 4
args.activation = 'Tanh'
args.seed = 0
args.timepoints = [0.5, 1.0, 1.5]
os.path.exists(args.input_dir)


if __name__ == '__main__':
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda:' + str(args.gpu)
                            if torch.cuda.is_available() else 'cpu')
    # load dataset
    data_train = loaddata(args,device)
    integral_time = args.timepoints

    time_pts = range(len(data_train))
    leave_1_out = []
    train_time = [x for i,x in enumerate(time_pts) if i!=leave_1_out]


    # model
    func = UOT(in_out_dim=data_train[0].shape[1], hidden_dim=args.hidden_dim,n_hiddens=args.n_hiddens,activation=args.activation).to(device)
    func.apply(initialize_weights)


    # configure training options
    options = {}
    options.update({'method': 'Dopri5'})
    options.update({'h': None})
    options.update({'rtol': 1e-3})
    options.update({'atol': 1e-5})
    options.update({'print_neval': False})
    options.update({'neval_max': 1000000})
    options.update({'safety': None})

    optimizer = optim.Adam(func.parameters(), lr=args.lr, weight_decay= 0.01)
    lr_adjust = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.niters-400,args.niters-200], gamma=0.5, last_epoch=-1)
    mse = nn.MSELoss()

    LOSS = []
    L2_1 = []
    L2_2 = []
    Trans = []
    Sigma = []
    
    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        ckpt_path = os.path.join(args.save_dir, 'ckpt.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            func.load_state_dict(checkpoint['func_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Loaded ckpt from {}'.format(ckpt_path))

    try:
        sigma_now = 1
        for itr in range(1, args.niters + 1):
            optimizer.zero_grad()
            
            loss, loss1, sigma_now, L2_value1, L2_value2 = train_model(mse,func,args,data_train,train_time,integral_time,sigma_now,options,device,itr)

            
            loss.backward()
            optimizer.step()
            lr_adjust.step()

            LOSS.append(loss.item())
            Trans.append(loss1[-1].mean(0).item())
            Sigma.append(sigma_now)
            L2_1.append(L2_value1.tolist())
            L2_2.append(L2_value2.tolist())
            
            print('Iter: {}, loss: {:.4f}'.format(itr, loss.item()))
            
            
            if itr % 20 == 0:
                ckpt_path = os.path.join(args.save_dir, 'ckpt_itr{}.pth'.format(itr))
                torch.save({'func_state_dict': func.state_dict()}, ckpt_path)
                print('Iter {}, Stored ckpt at {}'.format(itr, ckpt_path))
                
            
            

    except KeyboardInterrupt:
        if args.save_dir is not None:
            ckpt_path = os.path.join(args.save_dir, 'ckpt.pth')
            torch.save({
                'func_state_dict': func.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print('Stored ckpt at {}'.format(ckpt_path))
    print('Training complete after {} iters.'.format(itr))
    
    
    ckpt_path = os.path.join(args.save_dir, 'ckpt.pth')
    torch.save({
        'func_state_dict': func.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'LOSS':LOSS,
        'TRANS':Trans,
        'L2_1': L2_1,
        'L2_2': L2_2,
        'Sigma': Sigma
    }, ckpt_path)
    print('Stored ckpt at {}'.format(ckpt_path))