"""
PRESCIENT Data Preparation Script
==================================
Converts an h5ad file into a PRESCIENT-ready data.pt file:
  1. Splits train (Well != 2) / test (Well == 2)
  2. Exports train expression + metadata CSVs
  3. Calls PRESCIENT process_data (as Python import, not subprocess)
  4. Replaces data['xp'] with user-specified obsm embeddings
  5. Saves test cell embeddings / timepoints / cell types for evaluation

Usage
-----
python scripts/prescient/01_prepare_data.py \
    --data_path data/klein_subset.h5ad \
    --growth_path Explore_Notebook/8.Benchmark_v/Msig_GOBP_cellcycle_growth.pt \
    --obsm_key X_pca --n_dims 30 \
    --run_name pca_run

python scripts/prescient/01_prepare_data.py \
    --data_path data/klein_subset.h5ad \
    --growth_path Explore_Notebook/8.Benchmark_v/Msig_GOBP_cellcycle_growth.pt \
    --obsm_key DM_EigenVectors --n_dims 10 \
    --run_name dm_run
"""

import os
import argparse
import glob
import logging

import numpy as np
import pandas as pd
import torch
import scanpy as sc

# ── PyTorch ≥ 2.6 compat ─────────────────────────────────────────────────
# PRESCIENT internally uses torch.load() without weights_only=False,
# which fails on numpy-containing .pt files in PyTorch >= 2.6.
# Monkey-patch so all torch.load calls within this process default to
# weights_only=False (safe here since we trust PRESCIENT's own files).
_original_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(f, *args, **kwargs)
torch.load = _patched_torch_load

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare PRESCIENT data.pt from h5ad (with obsm replacement)"
    )
    p.add_argument("--data_path", required=True, help="Path to input .h5ad file")
    p.add_argument(
        "--growth_path",
        required=True,
        help="Path to pre-computed growth weights .pt file",
    )
    p.add_argument(
        "--output_dir",
        default="logs/PRESCIENT",
        help="Root output directory (default: logs/PRESCIENT/)",
    )
    p.add_argument(
        "--run_name",
        default=None,
        help="Experiment name → output saved to output_dir/run_name/",
    )
    p.add_argument(
        "--obsm_key",
        default="X_pca",
        choices=["X_pca", "DM_EigenVectors"],
        help="obsm key to use as cell embedding (default: X_pca)",
    )
    p.add_argument(
        "--n_dims",
        type=int,
        default=30,
        help="Number of embedding dimensions to retain (default: 30)",
    )
    p.add_argument(
        "--tp_col",
        default="timepoint_tx_days",
        help="Timepoint column in adata.obs (default: timepoint_tx_days)",
    )
    p.add_argument(
        "--well_col",
        default="Well",
        help="Well column for train/test split (default: Well)",
    )
    p.add_argument(
        "--celltype_col",
        default="label_man",
        help="Cell-type annotation column in adata.obs (default: anno_man)",
    )
    p.add_argument(
        "--num_pcs",
        type=int,
        default=50,
        help="PCs for PRESCIENT internal PCA step (default: 50; overridden by obsm replacement)",
    )
    return p.parse_args()


def run_prescient_process_data(expr_csv, meta_csv, growth_path, out_dir, tp_col, celltype_col, num_pcs):
    """
    Call PRESCIENT's process_data as a Python function (not subprocess).
    This lets the torch.load monkey-patch take effect.
    """
    from argparse import Namespace
    from prescient.commands.process_data import main as prescient_process_data

    prescient_args = Namespace(
        data_path=expr_csv,
        meta_path=meta_csv,
        out_dir=out_dir,
        tp_col=tp_col,
        celltype_col=celltype_col,
        num_pcs=num_pcs,
        num_neighbors_umap=10,
        growth_path=growth_path,
    )
    log.info(f"Calling prescient.commands.process_data.main() with args:")
    for k, v in vars(prescient_args).items():
        log.info(f"  {k} = {v}")

    prescient_process_data(prescient_args)


def main():
    args = parse_args()

    # ------------------------------------------------------------------ paths
    out_dir = args.output_dir
    if args.run_name:
        out_dir = os.path.join(out_dir, args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------ load
    log.info(f"Loading adata from {args.data_path}")
    adata = sc.read_h5ad(args.data_path)
    log.info(f"  Total cells: {adata.n_obs}  genes: {adata.n_vars}")

    # ------------------------------------------------------------------ split
    if args.well_col not in adata.obs.columns:
        raise ValueError(
            f"Column '{args.well_col}' not found in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )
    train_mask = adata.obs[args.well_col] != 2
    test_mask = adata.obs[args.well_col] == 2
    adata_train = adata[train_mask]
    adata_test = adata[test_mask]
    log.info(
        f"  Train cells: {train_mask.sum()}  Test cells: {test_mask.sum()}"
    )

    # ------------------------------------------------------------------ export CSVs
    expr_csv = os.path.join(out_dir, "train_expr.csv")
    meta_csv = os.path.join(out_dir, "train_meta.csv")

    log.info(f"Exporting expression matrix → {expr_csv}")
    import scipy.sparse
    X = adata_train.X
    if scipy.sparse.issparse(X):
        X = X.toarray()
    expr_df = pd.DataFrame(X, index=adata_train.obs_names, columns=adata_train.var_names)
    expr_df.to_csv(expr_csv)

    log.info(f"Exporting metadata → {meta_csv}")
    meta_df = adata_train.obs[[args.tp_col, args.celltype_col]].copy()
    meta_df.to_csv(meta_csv)

    # ------------------------------------------------------------------ prescient process_data
    # Call as Python import (not subprocess) so torch.load patch takes effect
    run_prescient_process_data(
        expr_csv=expr_csv,
        meta_csv=meta_csv,
        growth_path=args.growth_path,
        out_dir=out_dir,
        tp_col=args.tp_col,
        celltype_col=args.celltype_col,
        num_pcs=args.num_pcs,
    )

    # ------------------------------------------------------------------ find output data.pt
    # PRESCIENT names the file based on input CSV stem and growth weight stem
    candidates = sorted(glob.glob(os.path.join(out_dir, "*data.pt")))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(out_dir, "data.pt")))
    if not candidates:
        raise FileNotFoundError(
            f"No *data.pt found in {out_dir} after running prescient process_data. "
            f"Contents: {os.listdir(out_dir)}"
        )
    # Pick the most recently modified one
    data_pt_path = max(candidates, key=os.path.getmtime)
    log.info(f"Found data.pt at: {data_pt_path}")

    # ------------------------------------------------------------------ replace xp
    log.info(f"Loading data.pt and replacing xp with {args.obsm_key}[:{args.n_dims}]")
    data = torch.load(data_pt_path, weights_only=False)
    log.info(f"  data.pt keys: {list(data.keys())}")
    log.info(f"  Timepoints: {data['y']}")

    # data['xp'] is a list of tensors, one per timepoint
    # Build matching replacement from train cells
    tps_unique = data["y"]
    tp_values = sorted(adata_train.obs[args.tp_col].unique())
    if len(tp_values) != len(tps_unique):
        log.warning(
            f"Number of timepoints in adata ({len(tp_values)}) differs "
            f"from data.pt ({len(tps_unique)}). Using adata order."
        )

    obsm_key = args.obsm_key
    if obsm_key not in adata_train.obsm:
        raise KeyError(
            f"obsm key '{obsm_key}' not found. Available: {list(adata_train.obsm.keys())}"
        )

    new_xp = []
    for tp in tp_values:
        mask_tp = adata_train.obs[args.tp_col] == tp
        emb = adata_train[mask_tp].obsm[obsm_key][:, : args.n_dims].astype(np.float32)
        new_xp.append(torch.tensor(emb))
        log.info(f"    tp={tp}: {emb.shape}")

    data["xp"] = new_xp
    torch.save(data, data_pt_path)
    log.info(f"Saved modified data.pt → {data_pt_path}")
    log.info(f"  xp shapes: {[x.shape for x in data['xp']]}")

    # ------------------------------------------------------------------ save test cells
    test_emb = adata_test.obsm[obsm_key][:, : args.n_dims].astype(np.float32)
    test_tps = adata_test.obs[args.tp_col].values.astype(np.float32)
    test_ct = adata_test.obs[args.celltype_col].values.astype(str)

    np.save(os.path.join(out_dir, "test_cells.npy"), test_emb)
    np.save(os.path.join(out_dir, "test_tps.npy"), test_tps)
    np.save(os.path.join(out_dir, "test_celltypes.npy"), test_ct)

    log.info(f"Saved test_cells.npy  shape={test_emb.shape}")
    log.info(f"Saved test_tps.npy    shape={test_tps.shape}")
    log.info(f"Saved test_celltypes.npy  shape={test_ct.shape}")
    log.info("Done.")


if __name__ == "__main__":
    main()
