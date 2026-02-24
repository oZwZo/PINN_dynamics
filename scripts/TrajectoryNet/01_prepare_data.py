#!/rds/user/wz369/hpc-work/LIBS/mamba/envs/mioflow/bin/python
"""
TrajectoryNet Data Preparation Script
=======================================
Converts an h5ad file into TrajectoryNet-ready NPZ format:
  1. Splits train (Well != 2) / test (Well == 2)
  2. Standardizes embeddings (zero-mean, unit-variance) using train stats
  3. Maps timepoints [2,4,6] -> [0,1,2] as consecutive integers
  4. Saves NPZ with {obsm_key: embedding, sample_labels: int_labels}
  5. Saves scaler.npz, test_cells.npz, train_meta.npz, config.json

Note: We pre-standardize and do NOT pass --whiten to TrajectoryNet.

Usage
-----
python scripts/TrajectoryNet/01_prepare_data.py \
    --data_path data/klein/klein_addpop.h5ad \
    --output_dir logs/TrajectoryNet --run_name pca30 \
    --obsm_key X_pca --n_dims 30

python scripts/TrajectoryNet/01_prepare_data.py \
    --data_path data/klein/klein_addpop.h5ad \
    --output_dir logs/TrajectoryNet --run_name dm10 \
    --obsm_key DM_EigenVectors --n_dims 10
"""

import os
import sys
import json
import argparse
import logging

import numpy as np
import scanpy as sc

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare TrajectoryNet NPZ data from h5ad (with standardization)"
    )
    p.add_argument("--data_path", required=True, help="Path to input .h5ad file")
    p.add_argument("--output_dir", default="logs/TrajectoryNet",
                   help="Root output directory (default: logs/TrajectoryNet)")
    p.add_argument("--run_name", default=None,
                   help="Experiment name -> output saved to output_dir/run_name/")
    p.add_argument("--obsm_key", default="X_pca",
                   help="obsm key for cell embeddings (default: X_pca)")
    p.add_argument("--n_dims", type=int, default=30,
                   help="Number of embedding dimensions to retain (default: 30)")
    p.add_argument("--tp_col", default="timepoint_tx_days",
                   help="Timepoint column in adata.obs (default: timepoint_tx_days)")
    p.add_argument("--well_col", default="Well",
                   help="Well column for train/test split (default: Well)")
    p.add_argument("--celltype_col", default="label_man",
                   help="Cell-type annotation column (default: label_man)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    return p.parse_args()


def main():
    args = parse_args()

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
    train_mask = adata.obs[args.well_col] != 2
    test_mask = adata.obs[args.well_col] == 2
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

    # ── map timepoints to consecutive integers ──
    train_tps = adata_train.obs[args.tp_col].values.astype(float)
    train_cts = adata_train.obs[args.celltype_col].values.astype(str)
    tp_values = sorted(np.unique(train_tps))
    log.info(f"  Original timepoints: {tp_values}")

    # normalize: [2.0, 4.0, 6.0] -> [0, 1, 2]
    tp_min = tp_values[0]
    tp_diffs = [tp_values[i+1] - tp_values[i] for i in range(len(tp_values)-1)]
    tp_step = tp_diffs[0]
    for d in tp_diffs[1:]:
        tp_step = min(tp_step, d)
    normalized_tps = [(tp - tp_min) / tp_step for tp in tp_values]
    log.info(f"  Normalized timepoints: {normalized_tps}")

    # Map each cell's timepoint to the integer label
    train_int_labels = np.array([(t - tp_min) / tp_step for t in train_tps], dtype=int)

    # ── build TrajectoryNet NPZ ──
    # TrajectoryNet CustomData expects: {embedding_name: array, sample_labels: int_array}
    npz_path = os.path.join(out_dir, "klein_train.npz")
    np.savez(
        npz_path,
        **{obsm_key: train_emb.astype(np.float32)},
        sample_labels=train_int_labels,
    )
    log.info(f"  Saved TrajectoryNet NPZ -> {npz_path}")
    log.info(f"    {obsm_key} shape: {train_emb.shape}")
    log.info(f"    sample_labels unique: {np.unique(train_int_labels)}")
    for lbl in np.unique(train_int_labels):
        n = np.sum(train_int_labels == lbl)
        log.info(f"    label={lbl}: {n} cells")

    # ── save train metadata (per-timepoint, for KNN in evaluation) ──
    n_tp = len(tp_values)
    train_meta_embs = []
    train_meta_cts = []
    for i, tp in enumerate(tp_values):
        tp_mask = train_tps == tp
        tp_emb = train_emb[tp_mask].astype(np.float32)
        tp_ct = train_cts[tp_mask]
        train_meta_embs.append(tp_emb)
        train_meta_cts.append(tp_ct)
        log.info(f"  tp={tp} (norm={normalized_tps[i]}): {tp_emb.shape[0]} cells")

    train_meta_path = os.path.join(out_dir, "train_meta.npz")
    meta_embs = np.empty(n_tp, dtype=object)
    meta_cts = np.empty(n_tp, dtype=object)
    for i in range(n_tp):
        meta_embs[i] = train_meta_embs[i]
        meta_cts[i] = train_meta_cts[i]
    np.savez(train_meta_path, embeddings=meta_embs, celltypes=meta_cts)
    log.info(f"  Saved train metadata -> {train_meta_path}")

    # ── save test cells ──
    test_tps = adata_test.obs[args.tp_col].values.astype(float)
    test_cts = adata_test.obs[args.celltype_col].values.astype(str)
    test_tps_norm = (test_tps - tp_min) / tp_step

    test_path = os.path.join(out_dir, "test_cells.npz")
    np.savez(test_path,
             embeddings=test_emb.astype(np.float32),
             timepoints=test_tps_norm.astype(np.float32),
             timepoints_orig=test_tps.astype(np.float32),
             celltypes=test_cts)
    log.info(f"  Saved test cells -> {test_path}  ({test_emb.shape[0]} cells)")

    # ── save config ──
    config = {
        "obsm_key": obsm_key,
        "n_dims": n_dims,
        "original_timepoints": tp_values,
        "normalized_timepoints": normalized_tps,
        "tp_min": tp_min,
        "tp_step": tp_step,
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
