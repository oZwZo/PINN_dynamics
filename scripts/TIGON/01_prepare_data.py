#!/rds/user/wz369/hpc-work/LIBS/mamba/envs/mioflow/bin/python
"""
TIGON Data Preparation Script
==============================
Converts an h5ad file into TIGON-ready .npy format:
  1. Splits train (Well != 2) / test (Well == 2)
  2. Standardizes embeddings (zero-mean, unit-variance) using train stats
  3. Subsamples train cells per timepoint (stratified by cell type)
  4. Normalizes timepoints [2,4,6] -> [0,1,2]
  5. Saves TIGON-format .npy, scaler, test cells, train metadata

Usage
-----
python scripts/TIGON/01_prepare_data.py \
    --data_path data/klein_addpop.h5ad \
    --output_dir logs/TIGON --run_name pca \
    --obsm_key X_pca --n_dims 30 --subsample 2000

python scripts/TIGON/01_prepare_data.py \
    --data_path data/klein_addpop.h5ad \
    --output_dir logs/TIGON --run_name dm \
    --obsm_key DM_EigenVectors --n_dims 10 --subsample 2000
"""

import os
import sys
import json
import argparse
import logging

import numpy as np
import scanpy as sc

sys.path.insert(0, '/rds/user/wz369/hpc-work/TIGON')

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare TIGON .npy data from h5ad (with standardization + subsampling)"
    )
    p.add_argument("--data_path", required=True, help="Path to input .h5ad file")
    p.add_argument("--output_dir", default="logs/TIGON",
                   help="Root output directory (default: logs/TIGON)")
    p.add_argument("--run_name", default=None,
                   help="Experiment name -> output saved to output_dir/run_name/")
    p.add_argument("--obsm_key", default="X_pca",
                   help="obsm key for cell embeddings (default: X_pca)")
    p.add_argument("--n_dims", type=int, default=30,
                   help="Number of embedding dimensions to retain (default: 30)")
    p.add_argument("--tp_col", default="timepoint_tx_days",
                   help="Timepoint column in adata.obs (default: timepoint_tx_days)")
    p.add_argument("--well_col", default="Well",
                   help="Column used for train/test split (default: Well). "
                        "For cord blood, pass 'split'.")
    p.add_argument("--test_values", nargs="+", default=["2"],
                   help="Values of --well_col that mark held-out test cells "
                        "(string-compared after .astype(str)). Default ['2'] "
                        "reproduces Klein behaviour; for cord blood pass 'test'.")
    p.add_argument("--dataset_name", default="klein_train",
                   help="Output npy stem (default 'klein_train'). TIGON 02_train.py "
                        "loads <input_dir>/<dataset>.npy where dataset = --dataset arg.")
    p.add_argument("--celltype_col", default="Annotation",
                   help="Cell-type annotation column (default: Annotation)")
    p.add_argument("--subsample", type=int, default=2000,
                   help="Max cells per timepoint (default: 2000, 0=no subsampling)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    return p.parse_args()


def stratified_subsample(embeddings, celltypes, n_target, rng):
    """Subsample to n_target cells, stratified by cell type."""
    n_total = len(celltypes)
    if n_target <= 0 or n_total <= n_target:
        return np.arange(n_total)

    unique_types = np.unique(celltypes)
    indices = []
    for ct in unique_types:
        ct_idx = np.where(celltypes == ct)[0]
        # proportional allocation
        n_ct = max(1, int(round(len(ct_idx) / n_total * n_target)))
        if n_ct >= len(ct_idx):
            indices.append(ct_idx)
        else:
            indices.append(rng.choice(ct_idx, size=n_ct, replace=False))

    indices = np.concatenate(indices)
    # if we overshot or undershot, adjust
    if len(indices) > n_target:
        indices = rng.choice(indices, size=n_target, replace=False)
    return np.sort(indices)


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    # ── output directory ──
    out_dir = args.output_dir
    if args.run_name:
        out_dir = os.path.join(out_dir, args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── load data ──
    log.info(f"Loading adata from {args.data_path}")
    adata = sc.read_h5ad(args.data_path)
    log.info(f"  Total cells: {adata.n_obs}  genes: {adata.n_vars}")

    # ── train/test split ──
    if args.well_col not in adata.obs.columns:
        raise ValueError(
            f"Column '{args.well_col}' not found. Available: {list(adata.obs.columns)}"
        )
    test_set = {str(v) for v in args.test_values}
    well_str = adata.obs[args.well_col].astype(str)
    test_mask = well_str.isin(test_set).values
    train_mask = ~test_mask
    if not test_mask.any():
        raise ValueError(
            f"No cells matched test_values={sorted(test_set)} in column "
            f"'{args.well_col}'. Unique values seen: "
            f"{sorted(well_str.unique().tolist())[:10]}..."
        )
    adata_train = adata[train_mask].copy()
    adata_test = adata[test_mask].copy()
    log.info(f"  Train cells: {adata_train.n_obs}  Test cells: {adata_test.n_obs}")

    # ── extract embeddings ──
    obsm_key = args.obsm_key
    n_dims = args.n_dims
    if obsm_key not in adata_train.obsm:
        raise KeyError(
            f"obsm key '{obsm_key}' not found. Available: {list(adata_train.obsm.keys())}"
        )

    train_emb = adata_train.obsm[obsm_key][:, :n_dims].astype(np.float64)
    test_emb = adata_test.obsm[obsm_key][:, :n_dims].astype(np.float64)
    log.info(f"  Embedding shape: train={train_emb.shape}  test={test_emb.shape}")

    # ── standardize using train statistics ──
    train_mean = train_emb.mean(axis=0)
    train_std = train_emb.std(axis=0)
    train_std = np.clip(train_std, 1e-6, None)  # clip near-zero variance dims

    train_emb = (train_emb - train_mean) / train_std
    test_emb = (test_emb - train_mean) / train_std

    log.info(f"  Standardized: train range [{train_emb.min():.2f}, {train_emb.max():.2f}]")
    log.info(f"  Standardized: test  range [{test_emb.min():.2f}, {test_emb.max():.2f}]")

    # ── save scaler ──
    scaler_path = os.path.join(out_dir, "scaler.npz")
    np.savez(scaler_path, mean=train_mean, std=train_std)
    log.info(f"  Saved scaler -> {scaler_path}")

    # ── group train cells by timepoint ──
    train_tps = adata_train.obs[args.tp_col].values.astype(float)
    train_cts = adata_train.obs[args.celltype_col].values.astype(str)
    tp_values = sorted(np.unique(train_tps))
    log.info(f"  Original timepoints: {tp_values}")

    # normalize timepoints: subtract min, divide by gcd of differences
    tp_min = tp_values[0]
    tp_diffs = [tp_values[i+1] - tp_values[i] for i in range(len(tp_values)-1)]
    tp_step = tp_diffs[0]
    for d in tp_diffs[1:]:
        # find gcd of float diffs (use integer trick)
        tp_step = min(tp_step, d)
    normalized_tps = [(tp - tp_min) / tp_step for tp in tp_values]
    log.info(f"  Normalized timepoints: {normalized_tps}")

    # ── subsample and build TIGON arrays ──
    data_arrays = []  # one array per timepoint
    train_meta_embs = []  # for KNN in evaluation
    train_meta_cts = []

    for i, tp in enumerate(tp_values):
        tp_mask = train_tps == tp
        tp_emb = train_emb[tp_mask]
        tp_ct = train_cts[tp_mask]

        log.info(f"  tp={tp} (norm={normalized_tps[i]}): {tp_emb.shape[0]} cells")

        if args.subsample > 0 and tp_emb.shape[0] > args.subsample:
            sel_idx = stratified_subsample(tp_emb, tp_ct, args.subsample, rng)
            tp_emb_sub = tp_emb[sel_idx]
            tp_ct_sub = tp_ct[sel_idx]
            log.info(f"    Subsampled to {tp_emb_sub.shape[0]} cells (stratified)")
        else:
            tp_emb_sub = tp_emb
            tp_ct_sub = tp_ct

        data_arrays.append(tp_emb_sub.astype(np.float32))
        train_meta_embs.append(tp_emb_sub.astype(np.float32))
        train_meta_cts.append(tp_ct_sub)

    # ── save TIGON-format .npy ──
    # Format: data = np.empty((1, n_timepoints), dtype=object); data[0, i] = array
    n_tp = len(data_arrays)
    data = np.empty((1, n_tp), dtype=object)
    for i in range(n_tp):
        data[0, i] = data_arrays[i]

    # Filename derives from --dataset_name (default 'klein_train' for backwards
    # compat); cord blood passes --dataset_name cordblood so TIGON 02_train can find it.
    dataset_name = getattr(args, "dataset_name", None) or "klein_train"
    npy_path = os.path.join(out_dir, f"{dataset_name}.npy")
    np.save(npy_path, data)
    log.info(f"  Saved TIGON data -> {npy_path}")
    for i in range(n_tp):
        log.info(f"    data[0,{i}] shape: {data[0, i].shape}")

    # ── save test cells ──
    test_tps = adata_test.obs[args.tp_col].values.astype(float)
    test_cts = adata_test.obs[args.celltype_col].values.astype(str)
    # normalize test timepoints using same scheme
    test_tps_norm = (test_tps - tp_min) / tp_step

    test_path = os.path.join(out_dir, "test_cells.npz")
    np.savez(test_path,
             embeddings=test_emb.astype(np.float32),
             timepoints=test_tps_norm.astype(np.float32),
             timepoints_orig=test_tps.astype(np.float32),
             celltypes=test_cts)
    log.info(f"  Saved test cells -> {test_path}  ({test_emb.shape[0]} cells)")

    # ── save train metadata (for KNN in evaluation) ──
    train_meta_path = os.path.join(out_dir, "train_meta.npz")
    # save as object arrays since they have different sizes per timepoint
    meta_embs = np.empty(n_tp, dtype=object)
    meta_cts = np.empty(n_tp, dtype=object)
    for i in range(n_tp):
        meta_embs[i] = train_meta_embs[i]
        meta_cts[i] = train_meta_cts[i]
    np.savez(train_meta_path, embeddings=meta_embs, celltypes=meta_cts)
    log.info(f"  Saved train metadata -> {train_meta_path}")

    # ── save config ──
    config = {
        "obsm_key": obsm_key,
        "n_dims": n_dims,
        "original_timepoints": tp_values,
        "normalized_timepoints": normalized_tps,
        "tp_min": tp_min,
        "tp_step": tp_step,
        "subsample": args.subsample,
        "seed": args.seed,
        "in_out_dim": n_dims,
        "well_col": args.well_col,
        "tp_col": args.tp_col,
        "celltype_col": args.celltype_col,
    }
    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    log.info(f"  Saved config -> {config_path}")

    log.info("Done.")


if __name__ == "__main__":
    main()
