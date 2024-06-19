import pandas as pd
import numpy as np
from scipy.stats import cumfreq
import torch

# Reading the CSV files

# cluster 7 meta
R = pd.read_csv("/home/wergillius/Project/HSPCdynamics/PD_model/clu_7/tables/input_pseudo_dyn_clu_7_dpt.csv")
RN = pd.read_csv("/home/wergillius/Project/HSPCdynamics/PD_model/clu_7/tables/input_pseudo_dyn_clu_7_size.csv")

dpt_min = R.dpt_pseudotime.values.min()
dpt_max = R.dpt_pseudotime.values.max()
dpt_scaled = (R.dpt_pseudotime.values - dpt_min) / (dpt_max - dpt_min)
R['dpt_pseudotime'] = dpt_scaled
# ad = sc.read_h5ad(f"{data_path}/combined_filt.h5ad")

# Number of replicates for population size
n = RN.iloc[:, 3].values

# Population information
D = {}
D['pop'] = {}
D['pop']['t'] = RN.iloc[:, 0].values
D['pop']['mean'] = RN.iloc[:, 1].values
D['pop']['var'] = np.square(RN.iloc[:, 2].values) / n

# Scaling to 0.9 (omitted for simulated data)
# R.iloc[:, 4] = R.iloc[:, 4] * 0.9 / max(R.iloc[:, 4])

Rbatch = R.iloc[:, 1].values
Rulabels, ia, ic = np.unique(Rbatch, return_index=True, return_inverse=True)
tp = R.iloc[ia, 0].values
tp, indT = np.sort(tp), np.argsort(tp)
Rulabels = Rulabels[indT]
D_tmp = R.iloc[:, 2].values   # diffusion pseudo-time

# ['Var2']
indE = [np.where(R.iloc[:,1] == label)[0] for label in Rulabels]
D['ind'] = {'tp': [], 'hist': [], 'size': []}

for iL in range(len(Rulabels)):
    D['ind']['tp'].append(R.iloc[indE[iL][0], 0])
    D['ind']['hist'].append(D_tmp[indE[iL]])
    D['ind']['size'].append(len(indE[iL]))

# Area statistic
csd, xsd = [], []

# Empirical CDF for all replicates
for hist in D['ind']['hist']:
    res = cumfreq(hist, numbins=len(hist))
    freq = res.cumcount
    
    # x is actually pseudo-time range
    x = res.lowerlimit + np.linspace(0, res.binsize*res.cumcount.size,
                                 res.cumcount.size)
    csd.append(freq[1:] / len(hist))
    xsd.append(x[1:])

csdt, xsdt, D['xsdt'], D['csdt'] = [], [], [], []
areat = []
indr_len_it = []

# Empirical CDF for pooled data of branch 1
for t in D['pop']['t']:
    D_t = []              # pseudo-time (dpt) of cell from timepoint t
    
    # the index of the libarary that is collected at timepoint t
    ind_r = [i for i, tp_val in enumerate(D['ind']['tp']) if tp_val == t]
    indr_len_it.append(len(ind_r))

    for ir in ind_r:
        D_t.extend(D['ind']['hist'][ir])  # extend into one list
        
    res = cumfreq(D_t, numbins=len(D_t))
    freq = res.cumcount
    values = res.lowerlimit + np.linspace(0, res.binsize * res.cumcount.size,
                                 res.cumcount.size) 
    
    xsdt.append(values[1:])               # dpt value range
    csdt.append(freq[1:] / len(D_t))      # cumulative density of dpt
    
    
    csdr = torch.empty((len(xsdt[-1]), len(ind_r))) # the first dimension is interpolated
    area_temp = []
    
    for ih, ir in enumerate(ind_r):
        
        # evaluated at x-coordinate xsdt (all time t) based on per libaray value (xsd, csd)
        # the result is the interpolated cumulative density at xsdt
        csdr[:, ih] = torch.tensor(np.interp(xsdt[-1], xsd[ir], csd[ir]))
        area_temp.append(torch.sum(torch.diff(torch.from_numpy(xsdt[-1])) * torch.abs(torch.tensor(csdt[-1][:-1]) - csdr[:-1, ih])))
    areat.append(torch.tensor(area_temp))

D['xsdt'] = xsdt
D['csdt'] = csdt
    
areat = [torch.tensor(area) for area in areat]
# indr_len_it = torch.tensor(indr_len_it)
    
D['dist'] = {}
D['dist']['mean'] = torch.tensor([torch.mean(area) for area in areat])
D['dist']['var'] = torch.tensor([torch.var(area/indr_len_it[i]) for i,area in enumerate(areat)])
# torch.var( torch.div(areat, indr_len_it.view(-1,1)), dim=1)

torch.save(D, '../data/HSPC_clu7.pt')
