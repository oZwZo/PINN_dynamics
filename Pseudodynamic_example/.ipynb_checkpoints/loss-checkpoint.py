import numpy as np
import torch
from scipy.stats import gaussian_kde
from scipy.integrate import trapz
from scipy.stats import entropy



def llPseudodynamicsFvKS(theta, modelfun, D, options=None):
    """Implement your likelihood function here"""
    
    
    n_grid = 300
   
    if options is None:
        options = {'grid': np.linspace(0, 1, n_grid),
               'alpha': 100,
               'n_exp': 1}
    else:
        options = options

    x = options['grid']
    
    # u0 approximated by kernel density estimate
    if 'u0' not in options:
        u0 = np.zeros((3, n_grid))
        N01 = np.zeros(3)
        for i0 in range(3):
            density = gaussian_kde(D['ind']['hist'][i0])
            u0[i0, :] = density(x)
            u0[i0, :] /= trapz(u0[i0, :], x)
            N01[i0] = len(D['ind']['hist'][i0])

        options['u0'] = np.mean(u0, axis=0) * D['pop']['mean'][0]
        options['u0'] = 0.5 * (options['u0'][:-1] + options['u0'][1:])
    else:
        u0 = options['u0']

    options_sim = {'sensi': 1,
                   'maxsteps': 1e6,
                   'sx0': np.zeros((n_grid - 1, len(theta)))}

    # Simulation
    status, t_sim, _, u, _, us = modelFun(D['pop']['t'], theta, u0, [], options_sim)

    if status != 0:
        raise ValueError('AMICI simulation failed')
    
    # 
    N = u[:, -1]
    dNdtheta = us[:, -1, :]

    u = u[:, :-1]
    dudtheta = us[:, :-1, :]

    # Computation of probability
    p = u / N.reshape(-1, 1) * (1 / (n_grid - 1))
    dpdtheta = 1 / (n_grid - 1) * (N.reshape(-1, 1) * dudtheta - u * dNdtheta) / N.reshape(-1, 1) ** 2

    # Compute simulated cdf
    cs = np.cumsum(p, axis=1)
    dcsdtheta = np.cumsum(dpdtheta, axis=1)

    # Log-likelihood
    LogL = 0
    grad = np.zeros_like(theta)

    area = np.zeros((len(D['pop']['t']) - 1, 1))
    dareadtheta = np.zeros((len(D['pop']['t']) - 1, len(theta)))

    # Area statistics
    for it in range(1, len(D['pop']['t'])):
        x_combined = options['x_combined'][it]
        Aug_matrix = options['Aug_matrix'][it]
        cs_a = np.dot(Aug_matrix, cs[it, :])
        dcs_adtheta = np.dot(Aug_matrix, dcsdtheta[it, :, :])
        csd_a = D['csd_a'][it]
        area[it - 1] = np.sum(np.diff(x_combined) * np.abs(cs_a[:-1] - csd_a[:-1]))
        dareadtheta[it - 1, :] = np.dot(np.diff(x_combined) * np.sign(cs_a[:-1] - csd_a[:-1]), dcs_adtheta[:-1, :])
        LogL += 0.5 * np.log(2 * np.pi * D['dist']['var'][it]) + 0.5 * ((D['dist']['mean'][it] - area[it - 1]) ** 2) / D['dist']['var'][it]
        grad -= ((D['dist']['mean'][it] - area[it - 1]) / D['dist']['var'][it]) * dareadtheta[it - 1, :]

    # Population level statistics
    for it in range(1, len(D['pop']['t'])):
        LogL += 0.5 * np.log(2 * np.pi * D['pop']['var'][it] / options['n_exp']) + \
                0.5 * ((D['pop']['mean'][it] - N[it]) ** 2) / (D['pop']['var'][it] / options['n_exp'])
        grad -= ((D['pop']['mean'][it] - N[it]) / (D['pop']['var'][it] / options['n_exp'])) * dNdtheta[it, :]

    # Regularization
    alpha = options['alpha']
    grad += alpha * 2 * np.concatenate((-(theta[1:9] - theta[:8]), np.array([0]),
                                        -(theta[10:18] - theta[9:17]), np.array([0]),
                                        -(theta[19:27] - theta[18:26]), np.array([0])))

    LogL += alpha * (np.sum((theta[1:9] - theta[:8]) ** 2) +
                     np.sum((theta[10:18] - theta[9:17]) ** 2) +
                     np.sum((theta[19:27] - theta[18:26]) ** 2))

    return LogL, grad
