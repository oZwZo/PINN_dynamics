"""
TIGON Combined Evaluation Script
===================================
Evaluates a trained TIGON model on held-out test cells.
Computes all metrics (fate accuracy, Pearson r, W2) in one run.

Cell extraction follows the pseudodynamics+ pattern: cells are matched
by barcode from adata (not from test_cells.npz), embeddings are
standardized before simulation, and inverse-standardized for W2.

Outputs:
  - eval_combined.csv        (one row per sim_mode)
  - F_hat_{mode}.csv         (per-cell fate predictions)
  - w2_per_clone_{mode}.csv  (per-clone W2 distances)

Usage
-----
python scripts/TIGON/03_evaluate.py \
    --data_dir logs/TIGON/pca \
    --checkpoint logs/TIGON/pca/model/ckpt.pth \
    --data_path data/klein_addpop.h5ad \
    --device cuda:0
"""

import os
import sys
import json
import argparse
import logging

import numpy as np
import pandas as pd
import scanpy as sc
import torch

sys.path.insert(0, '/rds/user/wz369/hpc-work/TIGON')
from utility import UOT

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import fate_eval_pipeline as fate_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def standardize(x, scaler):
    """Apply z-score standardization: (x - mean) / std."""
    if scaler is None:
        return x
    return (x - np.asarray(scaler['mean'])) / np.asarray(scaler['std'])


def parse_args():
    p = argparse.ArgumentParser(
        description="Combined evaluation for TIGON (fate accuracy + W2)"
    )
    p.add_argument("--data_dir", required=True,
                   help="Dir with scaler.npz, config.json, model checkpoint")
    p.add_argument("--checkpoint", required=True,
                   help="Path to TIGON .pth checkpoint")
    p.add_argument("--data_path", default="data/klein_addpop.h5ad",
                   help="Path to .h5ad file (for barcodes, clones, embeddings)")
    p.add_argument("--F_obs_path", default="data/klein/F_obs.csv",
                   help="Ground-truth fate proportions CSV")
    p.add_argument("--clone_proportions", default="data/klein/clone_proportions.csv",
                   help="Clone proportions CSV")
    p.add_argument("--celltype_col", default="Annotation",
                   help="Cell-type annotation column (default: Annotation)")
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--n_hiddens", type=int, default=4)
    p.add_argument("--activation", default="Tanh")
    p.add_argument("--n_sims_fate", type=int, default=100,
                   help="Trajectories per cell for fate (deterministic, so 1 used)")
    p.add_argument("--n_sims_w2", type=int, default=10,
                   help="Trajectories per cell for W2 (deterministic, so 1 used)")
    p.add_argument("--k", type=int, default=15, help="KNN neighbors")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default=None,
                   help="Output directory (default: data_dir/)")
    return p.parse_args()


def main():
    args = parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    log.info(f"Device: {device}")

    output_dir = args.output_dir or args.data_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── Load config ──
    config_path = os.path.join(args.data_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    in_out_dim = config["in_out_dim"]
    n_dims = in_out_dim
    norm_tps = config["normalized_timepoints"]
    obsm_key = config.get("obsm_key", "X_pca")
    log.info(f"Config: in_out_dim={in_out_dim}, obsm_key={obsm_key}, norm_tps={norm_tps}")

    # ── Load model ──
    func = UOT(in_out_dim=in_out_dim,
               hidden_dim=args.hidden_dim,
               n_hiddens=args.n_hiddens,
               activation=args.activation).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    func.load_state_dict(checkpoint['func_state_dict'])
    func.eval()
    log.info(f"Loaded model from {args.checkpoint}")

    # ── Load scaler ──
    scaler_path = os.path.join(args.data_dir, "scaler.npz")
    scaler = dict(np.load(scaler_path)) if os.path.exists(scaler_path) else None
    log.info(f"Scaler: {'loaded' if scaler else 'None'}")

    # ── Load adata ──
    log.info(f"Loading adata: {args.data_path}")
    adata = sc.read_h5ad(args.data_path)

    # ── Load ground truth ──
    F_obs = pd.read_csv(args.F_obs_path, index_col=0)
    F_obs.index = F_obs.index.astype(str)
    F_obs = F_obs[F_obs.index.isin(adata.obs_names.astype(str))]

    clone_proportions = pd.read_csv(args.clone_proportions, index_col=0)

    # ═══════════════════════════════════════════════════════════════════════
    # FATE EVALUATION — extract start cells from adata by F_obs barcodes
    # ═══════════════════════════════════════════════════════════════════════
    fobs_idx_str = F_obs.index.astype(str)
    valid_mask = adata.obs_names.astype(str).isin(fobs_idx_str)
    start_cells_raw = adata[valid_mask].obsm[obsm_key][:, :n_dims].astype(np.float32)
    start_barcodes = list(adata[valid_mask].obs_names.astype(str))
    start_cells_fate = standardize(start_cells_raw, scaler).astype(np.float32)
    log.info(f"Fate start cells: {len(start_cells_fate)} (matched from F_obs)")

    # Reference: ALL train cells (standardized) — matches pseudodynamics+ pattern
    adata_train = adata[adata.obs.Well != 2]
    tp_col = config.get("tp_col", "timepoint_tx_days")
    tp_values = sorted(adata_train.obs[tp_col].unique())
    last_tp = tp_values[-1]
    x_ref_raw = adata_train.obsm[obsm_key][:, :n_dims].astype(np.float32)
    x_ref = standardize(x_ref_raw, scaler).astype(np.float32)
    y_ref = adata_train.obs[args.celltype_col].values.astype(str)

    # Fate: simulate from norm_tp[0] to norm_tp[-1]
    t_start_fate = float(norm_tps[0])
    t_end_fate = float(norm_tps[-1])

    # ═══════════════════════════════════════════════════════════════════════
    # W2 EVALUATION — extract clone-filtered test cells from adata
    # ═══════════════════════════════════════════════════════════════════════
    test_ad = adata[adata.obs.Well == 2]
    w2_clone_mask = test_ad.obs.clones.isin(clone_proportions.index)
    w2_test_ad = test_ad[w2_clone_mask]

    src_ad_w2 = w2_test_ad[w2_test_ad.obs.timepoint_tx_days == 4]
    tgt_ad_w2 = w2_test_ad[w2_test_ad.obs.timepoint_tx_days == 6]

    # Extract raw embeddings then standardize for simulation
    src_cells_w2_raw = src_ad_w2.obsm[obsm_key][:, :n_dims].astype(np.float32)
    tgt_cells_w2_raw = tgt_ad_w2.obsm[obsm_key][:, :n_dims].astype(np.float32)
    src_cells_w2 = standardize(src_cells_w2_raw, scaler).astype(np.float32)
    tgt_cells_w2 = standardize(tgt_cells_w2_raw, scaler).astype(np.float32)

    src_clones_w2 = src_ad_w2.obs.clones.values
    tgt_clones_w2 = tgt_ad_w2.obs.clones.values

    log.info(f"W2 src (tp4): {len(src_cells_w2)}, tgt (tp6): {len(tgt_cells_w2)}")

    # ═══════════════════════════════════════════════════════════════════════
    # EVALUATE (ODE — deterministic)
    # ═══════════════════════════════════════════════════════════════════════
    mode = "ode"
    log.info(f"\n{'='*60}")
    log.info(f"Mode: {mode.upper()}")

    # ── Fate accuracy ──
    sim_fn_fate = fate_eval.make_tigon_sim_fn(t_start=t_start_fate, t_end=t_end_fate)

    fate_result = fate_eval.run_fate_evaluation(
        start_cells=start_cells_fate,
        start_cell_ids=start_barcodes,
        model=func,
        simulation_func=sim_fn_fate,
        F_obs=F_obs,
        x_ref=x_ref, y_ref=y_ref,
        n_sims=1,
        k=args.k,
        device=device,
    )
    log.info(f"  Fate accuracy: {fate_result['accuracy']:.4f}")
    log.info(f"  Fate Pearson r: {fate_result['pearson_r']:.4f}")
    fate_result['F_hat'].to_csv(os.path.join(output_dir, f"F_hat_{mode}.csv"))

    # ── Population W2 ──
    # Simulate tp4 → tp6 in standardized space, inverse-standardize for W2
    t_start_w2 = float(norm_tps[1])
    t_end_w2 = float(norm_tps[-1])
    sim_fn_w2 = fate_eval.make_tigon_sim_fn(t_start=t_start_w2, t_end=t_end_w2)

    w2_result = fate_eval.compute_w2_population(
        src_cells=src_cells_w2,
        target_cells=tgt_cells_w2,
        model=func,
        sim_fn=sim_fn_w2,
        n_sims=1,
        device=device,
        scaler=scaler,
    )
    log.info(f"  W2 scaled: {w2_result['w2_scaled']:.4f}")
    log.info(f"  W2 raw:    {w2_result['w2_raw']:.4f}")

    # ── Per-clone W2 ──
    df_clone_w2 = fate_eval.compute_w2_per_clone(
        src_cells=src_cells_w2,
        src_clone_ids=src_clones_w2,
        target_cells=tgt_cells_w2,
        target_clone_ids=tgt_clones_w2,
        model=func,
        sim_fn=sim_fn_w2,
        n_sims=1,
        device=device,
        scaler=scaler,
    )
    df_clone_w2.to_csv(os.path.join(output_dir, f"w2_per_clone_{mode}.csv"), index=False)
    log.info(f"  Per-clone W2: {len(df_clone_w2)} clones")

    # ── Save combined CSV ──
    df_combined = pd.DataFrame([{
        "sim_mode": mode,
        "accuracy": fate_result["accuracy"],
        "pearson_r": fate_result["pearson_r"],
        "w2_scaled": w2_result["w2_scaled"],
        "w2_raw": w2_result["w2_raw"],
        "n_start_cells": fate_result["n_start_cells"],
        "n_sims_fate": 1,
        "k": args.k,
    }])
    combined_path = os.path.join(output_dir, "eval_combined.csv")
    df_combined.to_csv(combined_path, index=False)
    log.info(f"\n{'='*60}")
    log.info(f"Combined results → {combined_path}")
    log.info(f"\n{df_combined.to_string(index=False)}")
    log.info("Done.")


if __name__ == "__main__":
    main()
