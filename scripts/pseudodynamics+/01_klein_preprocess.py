import sys, os
sys.path.insert(0, os.path.abspath("src"))

import scanpy as sc
import numpy as np
import pandas as pd
import pseudodynamics as pdp

DATA_PATH = "/home/wz369/rds/hpc-work/PINN_dynamics/data/klein/klein_addpop.h5ad"
OUT_PATH  = "data/klein_addpop.h5ad"

print("Loading data...")
adata = sc.read_h5ad(DATA_PATH)

# --- 1. Compute adata.uns['pop']['var'] ---
tp_col   = 'timepoint_tx_days'
well_col = 'Well'
timepoints = sorted(adata.obs[tp_col].unique())   # [2.0, 4.0, 6.0]

if 'var' not in adata.uns['pop']:
    adata.uns['pop']['var'] = adata.uns['pop']['std']**2

print("pop stats:", adata.uns['pop'])

# --- 2. Create obs['split'] using Well-based split ---
split = pd.Series('train', index=adata.obs_names)
test_mask = adata.obs[well_col].astype(int) == 2
split[test_mask] = 'test'

trainval_idx = adata.obs_names[~test_mask]
np.random.seed(42)
val_idx = np.random.choice(trainval_idx, size=int(len(trainval_idx) * 0.1), replace=False)
split[val_idx] = 'val'

adata.obs['split'] = split
print("Split distribution:\n", adata.obs['split'].value_counts())

# --- 3. Precompute Delta_DM (from DM_EigenVectors, 20 repeats) ---

if 'delta_DM' not in adata.obsm:
    print("Computing delta_DM...")
    deltaX_ts = []
    for i in range(20):
        delta_X, neighbor_ls = pdp.tl.sample_deltax(adata, xkey='DM_EigenVectors')
        deltaX_ts.append(delta_X)

    deltaX_ts = np.stack(deltaX_ts)  # (20, n_cells, 1, n_dims) — repeat=1 adds a dim
adata.obsm['delta_DM'] = deltaX_ts.mean(axis=0).squeeze(1)  # -> (n_cells, n_dims)
print("delta_DM shape:", adata.obsm['delta_DM'].shape)

# --- 4. Precompute Delta_PC (from X_pca, 20 repeats) ---
if 'delta_PC' not in adata.obsm:
    print("Computing delta_PC...")
    deltaX_ts = []
    for i in range(20):
        delta_X, neighbor_ls = pdp.tl.sample_deltax(adata, xkey='X_pca')
        deltaX_ts.append(delta_X)

    deltaX_ts = np.stack(deltaX_ts)  # (20, n_cells, 1, n_dims)
    adata.obsm['delta_PC'] = deltaX_ts.mean(axis=0).squeeze(1)  # -> (n_cells, n_dims)
print("delta_PC shape:", adata.obsm['delta_PC'].shape)

# --- 5. Save ---
os.makedirs("data", exist_ok=True)
adata.write_h5ad(OUT_PATH)
print(f"Saved to {OUT_PATH}")
