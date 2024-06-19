import pandas as pd
import numpy as np
from scipy.stats import cumfreq
import torch

# Reading the CSV files

# cluster 7 meta
R = pd.read_csv("/home/wergillius/Project/HSPCdynamics/PD_model/clu7_clu11/input_pseudo_dyn_ery_mk_dpt_feb_22.csv")
RN = pd.read_csv("/home/wergillius/Project/HSPCdynamics/PD_model/clu7_clu11/input_pseudo_dyn_ery_mk_size.csv")

celltype_col = 'dpt_adjusted_group'

celltypes = R[celltype_col].unique()
n_dim = len(celltypes) if 'progenitors' not in celltypes else len(celltypes) - 1

# this dictionary map string to dummy array
i = 0
cell_type_to_axis = dict()
for ct in celltypes:
    if ct == 'progenitors':
        cell_type_to_axis[ct] = np.ones(n_dim)
    else:
        ay = np.zeros(n_dim)
        ay[i] = 1
        cell_type_to_axis[ct] = ay
        i += 1

# we will get something like
# {    'mk' :          [1,0],
#     'ery':          [0,1],
#     'progenitors':  [1,1]}

dummy_map = np.stack(R[celltype_col].map(cell_type_to_axis))
dpt_array = R['dpt_pseudotime'].values.reshape(-1,1)    
dpt_highdim = np.multiply(dpt_array, dummy_map)  # assign dpt to each dimension

dpt_min = dpt_highdim.min(axis=0)
dpt_max = dpt_highdim.max(axis=0)
dpt_scaled = (dpt_highdim - dpt_min) / (dpt_max - dpt_min)
# R['dpt_pseudotime'] = dpt_scaled
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

Rbatch = R.loc[:, 'sample'].values
Rulabels, ia, ic = np.unique(Rbatch, return_index=True, return_inverse=True)
tp = R.iloc[ia]['stage'].values
tp, indT = np.sort(tp), np.argsort(tp)
Rulabels = Rulabels[indT]
D_tmp = dpt_highdim   # R.iloc[:, 2].values   # diffusion pseudo-time

# ['Var2']
indE = [np.where(R.loc[:, 'sample'].values == label)[0] for label in Rulabels]
D['ind'] = {'tp': [], 'hist': [], 'size': []}

for iL in range(len(Rulabels)):
    D['ind']['tp'].append(R.iloc[indE[iL][0]]['stage'])
    D['ind']['hist'].append(D_tmp[indE[iL]])
    D['ind']['size'].append(len(indE[iL]))


torch.save(D, '../data/HSPC_mk_ery.pt')
