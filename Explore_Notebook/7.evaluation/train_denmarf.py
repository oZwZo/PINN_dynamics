
import os, sys, re
import numpy as np
import pandas as pd
import scanpy as sc

from denmarf import DensityEstimate

main_dir="/ssd/users/Wergillius/Project/PINN_dynamics"
os.chdir(main_dir)


# argvs
if len(sys.argv) > 1:
    t = eval(sys.argv[1])

if len(sys.argv) > 2:
    device = f'cuda:{sys.argv[2]}'
else:
    device = 'cuda:0'

de = DensityEstimate(device=device)


# X is some np ndarray
data_name = 'gastrulation'
cellstate_key = 'DM_EigenVectors_multiscaled'
timepoint_key = 'timepoint_tx_days'

adata = sc.read_h5ad(f"data/{data_name}.h5ad")
ad_t = adata[adata.obs[timepoint_key] == t]

# so that all the X is scaled by the same std
if "multiscaled" in cellstate_key:
    X_scaled = ad_t.obsm[cellstate_key]
else:
    X = adata.obsm[cellstate_key]
    X_scaled = ad_t.obsm[cellstate_key] / X.std(axis=0).reshape(1,-1)


de = de.fit(X_scaled, bounded=True, lower_bounds=-60, upper_bounds=60, num_blocks=6)
# de = de.fit(X_scaled, bounded=False, num_blocks=6)


de.save(f"results/denmarf_model/{data_name}_{cellstate_key}_{t}.pkl")
print(f"{t} done training")