import numpy as np
import torch
from scipy.stats import gaussian_kde
from scipy.integrate import trapz
from scipy.stats import entropy
from scipy.optimize import minimize

# _ our own implementation
from functions import augment_cdf
from loss import llPseudodynamicsFvKS 


def mTe_fun(n):
    n = int(n)
    np.random.seed(10061988)

    # Model Definition
    n_grid = 300
    x = np.linspace(0, 1, n_grid)

    # Data
    ic = np.nan
    D = torch.load('dataExample.pt')
    

    # Determine regularization alpha depending on n
    if n < 81:
        alpha = 0
    elif n < 161:
        alpha = 1
    elif n < 241:
        alpha = 10

    options = {'alpha': alpha, 'Aug_matrix': [], 'x_combined': []}

    # Definition of the Parameter Estimation Problem
    model = 'finiteVolume'

    if model == 'finiteVolume':
        parameters = {
            'name': ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9',
                     'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9',
                     'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9'],
            'number': 27,
            'min': [-10.3616] * np.ones(9) + [-11.5129] * np.ones(9) + [-6] * np.ones(9),
            'max': [0] * np.ones(9) + [0] * np.ones(9) + [5] * np.ones(9),
            'guess': [-6.7] * np.ones(9) + [-2] * np.ones(9) + [-4] * 2 + [-10] + [-12] + [1.2] * np.ones(9)
        }
        modelfun = simulate_pd_fv
    elif model == 'branching_fv':
        parameters = {
            'name': ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12',
                     'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12',
                     'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'd12', 'd21'],
            'number': 36,
            'min': [-10.3616] * np.ones(12) + [-11.5129] * np.ones(12) + [-6] * np.ones(12) + [-10.3616, -10.3616],
            'max': [0] * np.ones(12) + [0] * np.ones(12) + [5] * np.ones(12) + [4.6052, 4.6052]
        }
        modelfun = simulate_pd_branching_fv2

    options['grid'] = x

    # Compute mean u0 distribution
    u0 = np.zeros((3, n_grid))
    N01 = np.zeros(3)
    for i0 in range(3):
        density = gaussian_kde(D['ind']['hist'][i0])
        u0[i0, :] = density(x)
        u0[i0, :] /= trapz(u0[i0, :], x)
        N01[i0] = len(D['ind']['hist'][i0])

    options['u0'] = np.mean(u0, axis=0) * D['pop']['mean'][0]
    options['u0'] = 0.5 * (options['u0'][:-1] + options['u0'][1:])

    options['n_exp'] = 1

    # Compute augmentation of data ecdfs to x_combined
    D['csd_a'] = []
    for it in range(6):  # 1 + len(D['pop']['t'])
        x_combined = np.union1d(x[:-1], D['xsdt'][it])
        D['csd_a'].append(
            augment_cdf(D['xsdt'][it], x_combined, D['csdt'][0].reshape(-1,1)))
        options['x_combined'].append(x_combined)
    
    # in matlab
    # D.csd_a{k1} = augment_cdf(D.xsdt{it},options.x_combined{k1},D.csdt{it});
    
    # Compute matrices for augmentation of cdfs

    ## original 
    # for it in range(1, len(D['pop']['t'])):
    #     options['Aug_matrix'][it] = np.zeros((n_grid - 1, n_grid - 1))
    #     for ig in range(n_grid - 1):
    #         e_i = np.zeros(n_grid - 1)
    #         e_i[ig] = 1
    #         options['Aug_matrix'][it][:, ig] = augment_cdf(options['grid'][:-1], x_combined, e_i) # ???
    
    ## debugged version
    for it in range(1, len(D['pop']['t'])):
        aug_M = []
        
        for ig in range(n_grid - 1):
            e_i = np.zeros((n_grid - 1,1))
            e_i[ig] = 1
            aug_M.append(  augment_cdf(x[:-1], options['x_combined'][it], e_i) ) # ???

        options['Aug_matrix'].append(np.hstack(aug_M))
                        

    # Log-likelihood function
    objectiveFunction = lambda theta: llPseudodynamicsFvKS(theta, modelfun, D, options)

    # Multi-start local optimization
    optionsMultistart = {
        'obj_type': 'negative log-posterior',
        'n_starts': 80,
        'comp_type': 'sequential',
        'mode': 'silent',
        'proposal': 'uniform',
        'localOptimizerOptions': {
            'Display': 'off',
            'Gradobj': 'on',
            'MaxIter': 6000,
            'MaxFunEvals': 12000
        },
        'start_index': (n - 1) % 80
    }

    # Optimization
    parameters = getMultiStarts(parameters, objectiveFunction, optionsMultistart)

    if n == 1 or n == 81 or n == 161:
        np.savez(f'parametersExample_{n}.npz', parameters=parameters, options=options, optionsMultistart=optionsMultistart)
    else:
        np.savez(f'parametersExample_{n}.npz', parameters=parameters)


def simulate_pd_fv(t, theta, u0, options_sim):
    pass  # Implement your simulation function here


def simulate_pd_branching_fv2(t, theta, u0, options_sim):
    pass  # Implement your simulation function here


def augment_cdf(x, x_a, y):
    pass  # Implement your cdf augmentation function here


def getMultiStarts(parameters, objectiveFunction, optionsMultistart):
    pass  # Implement your multi-start local optimization function here