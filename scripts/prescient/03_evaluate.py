"""
PRESCIENT Combined Evaluation Script
=======================================
Evaluates a trained PRESCIENT model on held-out test cells (Well == 2).
Computes all metrics (fate accuracy, Pearson r, W2) in one run.

Outputs:
  - eval_combined.csv        (one row per sim_mode)
  - F_hat_{mode}.csv         (per-cell fate predictions)
  - w2_per_clone_{mode}.csv  (per-clone W2 distances)

Usage
-----
python scripts/prescient/03_evaluate.py \
    --model_dir logs/PRESCIENT/pca_run/PCA-softplus_1_500-1e-06/seed_2 \
    --data_path data/klein_addpop.h5ad \
    --obsm_key X_pca --n_dims 30 --device cuda:0
"""

import os
import sys
import argparse
import logging

import numpy as np
import pandas as pd
import scanpy as sc
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import fate_eval_pipeline as fate_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Combined evaluation for PRESCIENT (fate accuracy + W2)"
    )
    p.add_argument("--model_dir", required=True,
                   help="PRESCIENT model dir with train.best.pt and config.pt")
    p.add_argument("--data_path", default="data/klein_addpop.h5ad",
                   help="Path to .h5ad file")
    p.add_argument("--F_obs_path", default="data/klein/F_obs.csv",
                   help="Ground-truth fate proportions CSV")
    p.add_argument("--clone_proportions", default="data/klein/clone_proportions.csv",
                   help="Clone proportions CSV")
    p.add_argument("--obsm_key", default="X_pca",
                   choices=["X_pca", "X_pca_scaled",
                            "X_pca_harmony", "X_pca_harmony_scaled",
                            "DM_EigenVectors", "DM_EigenVectors_scaled"],
                   help="obsm key used during training. *_scaled keys imply the "
                        "model lives in z-scored space; the evaluator pulls the "
                        "matching scaler from adata.uns to inverse to raw units.")
    p.add_argument("--n_dims", type=int, default=30,
                   help="Embedding dimensions")
    p.add_argument("--tp_col", default="timepoint_tx_days")
    p.add_argument("--well_col", default="Well")
    p.add_argument("--celltype_col", default="Annotation")
    p.add_argument("--n_sims_fate", type=int, default=100,
                   help="Trajectories per cell for fate accuracy")
    p.add_argument("--n_sims_w2", type=int, default=10,
                   help="Trajectories per cell for W2")
    p.add_argument("--k", type=int, default=15, help="KNN neighbors")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sim_sd", type=float, default=None,
                   help="Override config.train_sd at simulation time (no retraining).")
    p.add_argument("--output_dir", default=None,
                   help="Output directory (default: model_dir/, or "
                        "model_dir/eval_sim_sd_<value>/ when --sim_sd is set)")
    return p.parse_args()


def main():
    args = parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    log.info(f"Device: {device}")

    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.sim_sd is not None:
        output_dir = os.path.join(args.model_dir, f"eval_sim_sd_{args.sim_sd}")
    else:
        output_dir = args.model_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── Load model ──
    from prescient.train.model import SimpleNamespace, AutoGenerator

    config_path = os.path.join(args.model_dir, "config.pt")
    train_pt = os.path.join(args.model_dir, "train.best.pt")
    for p_path in (config_path, train_pt):
        if not os.path.exists(p_path):
            raise FileNotFoundError(f"Required file not found: {p_path}")

    config = SimpleNamespace(**torch.load(config_path, weights_only=False))
    if args.sim_sd is not None:
        log.info(
            f"Override sim noise: config.train_sd {config.train_sd} → {args.sim_sd}"
        )
        config.train_sd = args.sim_sd
    net = AutoGenerator(config)
    checkpoint = torch.load(train_pt, map_location=device, weights_only=False)
    net.load_state_dict(checkpoint["model_state_dict"])
    net = net.to(device).eval()
    log.info(f"Model: x_dim={config.x_dim}, layers={config.layers}, k_dim={config.k_dim}")

    # ── Load data ──
    log.info(f"Loading adata: {args.data_path}")
    adata = sc.read_h5ad(args.data_path)

    F_obs = pd.read_csv(args.F_obs_path, index_col=0)
    F_obs.index = F_obs.index.astype(str)

    clone_proportions = pd.read_csv(args.clone_proportions, index_col=0)

    train_mask = adata.obs[args.well_col] != 2
    test_mask = adata.obs[args.well_col] == 2

    tp_values = sorted(adata.obs[args.tp_col].unique())
    tp_first = tp_values[0]   # earliest timepoint (e.g., day 2 for klein, day 3 for cord blood)
    tp_mid = tp_values[1]     # middle timepoint
    tp_last = tp_values[-1]   # latest timepoint
    log.info(f"Timepoints: {tp_values}")

    # Scaler: use the refactored compute_scaler_from_adata, which reads
    # adata.uns['pca_scaler'] / adata.uns['dm_scaler'] when present (cord blood),
    # else falls back to on-the-fly fit (Klein).
    scaler = fate_eval.compute_scaler_from_adata(adata, args.obsm_key, args.n_dims)
    if scaler is not None:
        log.info(f"Scaler loaded for {args.obsm_key}: mean shape={scaler['mean'].shape}")
    else:
        log.info("Scaler: none → assuming model trained on raw obsm")

    # ── Prepare fate eval data ──
    fobs_idx = F_obs.index.astype(str)
    valid_mask = adata.obs_names.astype(str).isin(fobs_idx)
    tp_start_mask = (adata.obs[args.tp_col] == tp_first).values
    start_mask = valid_mask & tp_start_mask

    start_cells_fate = adata[start_mask].obsm[args.obsm_key][:, :args.n_dims].astype(np.float32)
    start_cell_ids = list(adata[start_mask].obs_names.astype(str))

    # KNN reference: training cells in the same coord system as the model
    x_ref = adata[train_mask].obsm[args.obsm_key][:, :args.n_dims].astype(np.float32)
    y_ref = adata[train_mask].obs[args.celltype_col].values.astype(str)

    # ── Prepare W2 data ──
    test_ad = adata[test_mask]
    w2_clone_mask = test_ad.obs.clones.isin(clone_proportions.index)
    w2_test_ad = test_ad[w2_clone_mask].copy()

    src_ad_w2 = w2_test_ad[w2_test_ad.obs[args.tp_col] == tp_mid]
    tgt_ad_w2 = w2_test_ad[w2_test_ad.obs[args.tp_col] == tp_last]

    src_cells_w2 = src_ad_w2.obsm[args.obsm_key][:, :args.n_dims].astype(np.float32)
    tgt_cells_w2 = tgt_ad_w2.obsm[args.obsm_key][:, :args.n_dims].astype(np.float32)
    src_clones_w2 = src_ad_w2.obs.clones.values
    tgt_clones_w2 = tgt_ad_w2.obs.clones.values

    log.info(f"Fate start cells: {len(start_cells_fate)}")
    log.info(f"W2 src (tp_mid): {len(src_cells_w2)}, tgt (tp_last): {len(tgt_cells_w2)}")

    # ── Evaluate (SDE only) ──
    mode = "sde"
    log.info(f"\n{'='*60}")
    log.info(f"Mode: {mode.upper()}")

    # Fate accuracy
    sim_fn_fate = fate_eval.make_prescient_sim_fn()
    fate_result = fate_eval.run_fate_evaluation(
        start_cells=start_cells_fate,
        start_cell_ids=start_cell_ids,
        model=(net, config),
        simulation_func=sim_fn_fate,
        F_obs=F_obs,
        x_ref=x_ref, y_ref=y_ref,
        n_sims=args.n_sims_fate,
        k=args.k,
        device=device,
    )
    log.info(f"  Fate accuracy: {fate_result['accuracy']:.4f}")
    log.info(f"  Fate Pearson r: {fate_result['pearson_r']:.4f}")
    fate_result['F_hat'].to_csv(os.path.join(output_dir, f"F_hat_{mode}.csv"))

    # Population W2
    w2_result = fate_eval.compute_w2_population(
        src_cells=src_cells_w2,
        target_cells=tgt_cells_w2,
        model=(net, config),
        sim_fn=sim_fn_fate,
        n_sims=args.n_sims_w2,
        device=device,
        scaler=scaler,
    )
    log.info(f"  W2 scaled: {w2_result['w2_scaled']:.4f}")
    log.info(f"  W2 raw:    {w2_result['w2_raw']:.4f}")

    # Per-clone W2
    df_clone_w2 = fate_eval.compute_w2_per_clone(
        src_cells=src_cells_w2,
        src_clone_ids=src_clones_w2,
        target_cells=tgt_cells_w2,
        target_clone_ids=tgt_clones_w2,
        model=(net, config),
        sim_fn=sim_fn_fate,
        n_sims=args.n_sims_w2,
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
        "n_sims_fate": args.n_sims_fate,
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
