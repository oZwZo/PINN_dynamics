"""
OT-CFM Combined Evaluation Script
====================================
Evaluates a trained OT-CFM model on held-out test cells.
Computes all metrics (fate accuracy, Pearson r, W2) in one run.

Outputs:
  - eval_combined.csv        (one row per sim_mode)
  - F_hat_{mode}.csv         (per-cell fate predictions)
  - w2_per_clone_{mode}.csv  (per-clone W2 distances)

Usage
-----
python scripts/otcfm/03_evaluate.py \
    --model_dir logs/otcfm/pca30/model \
    --data_path data/klein/klein_addpop.h5ad \
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

from torchcfm.models import MLP

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import fate_eval_pipeline as fate_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Combined evaluation for OT-CFM (fate accuracy + W2)"
    )
    p.add_argument("--model_dir", required=True,
                   help="Dir with ckpt.pt, config.json, scaler.npz, train_meta.npz, test_cells.npz")
    p.add_argument("--data_path", default="data/klein_addpop.h5ad",
                   help="Path to .h5ad file (must contain obsm_key used in training)")
    p.add_argument("--F_obs_path", default="data/klein/F_obs.csv",
                   help="Ground-truth fate proportions CSV")
    p.add_argument("--clone_proportions", default="data/klein/clone_proportions.csv",
                   help="Clone proportions CSV")
    p.add_argument("--celltype_col", default="Annotation")
    p.add_argument("--n_sims_fate", type=int, default=100,
                   help="Trajectories per cell for fate accuracy (deterministic, so 1 used)")
    p.add_argument("--n_sims_w2", type=int, default=10,
                   help="Trajectories per cell for W2 (deterministic, so 1 used)")
    p.add_argument("--k", type=int, default=15,
                   help="KNN neighbors")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default=None,
                   help="Output directory (default: model_dir/)")
    return p.parse_args()


def main():
    args = parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    log.info(f"Device: {device}")

    output_dir = args.output_dir or args.model_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── Load model ──
    config_path = os.path.join(args.model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    n_dims = config["n_dims"]
    width = config["width"]
    norm_tps = config["normalized_timepoints"]

    model = MLP(dim=n_dims, time_varying=True, w=width).to(device)
    ckpt = torch.load(os.path.join(args.model_dir, "ckpt.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info(f"Config: n_dims={n_dims}, norm_tps={norm_tps}")

    # ── Load npz data ──
    train_meta = np.load(os.path.join(args.model_dir, "train_meta.npz"), allow_pickle=True)
    train_embs = train_meta["embeddings"]
    train_cts = train_meta["celltypes"]

    test_data = np.load(os.path.join(args.model_dir, "test_cells.npz"), allow_pickle=True)
    test_emb = test_data["embeddings"]
    test_tps = test_data["timepoints"]
    test_barcodes = test_data["barcodes"] if "barcodes" in test_data else None

    # ── Load adata for clones ──
    log.info(f"Loading adata: {args.data_path}")
    adata = sc.read_h5ad(args.data_path)
    clone_proportions = pd.read_csv(args.clone_proportions, index_col=0)

    F_obs = pd.read_csv(args.F_obs_path, index_col=0)
    F_obs.index = F_obs.index.astype(str)

    # ── Prepare fate eval data (index adata day-first cells in F_obs, apply scaler) ──
    # Start cells can be in the training set, so pull them from adata (not test_cells.npz)
    fobs_idx_set = set(F_obs.index.astype(str))
    tp_col = "timepoint_tx_days"
    tp_values = sorted(adata.obs[tp_col].unique())
    tp_first_raw = tp_values[0]
    valid_mask = adata.obs_names.astype(str).isin(list(fobs_idx_set))
    tp_start_mask = (adata.obs[tp_col] == tp_first_raw).values
    start_mask = valid_mask & tp_start_mask
    start_adata = adata[start_mask]

    obsm_key = config.get("obsm_key", "X_pca_scaled")
    if obsm_key not in start_adata.obsm:
        fallback_key = "X_pca" if n_dims == 30 else "DM_EigenVectors"
        log.warning(f"obsm_key '{obsm_key}' not in adata; falling back to '{fallback_key}'")
        obsm_key = fallback_key

    # ── Load training-time scaler (the only valid inverse to raw PC space) ──
    # Training (02_train.py) standardizes raw obsm[X_pca] with TRAIN-ONLY mean/std
    # and saves them to scaler.npz. compute_scaler_from_adata would use full-pop
    # stats, which is NOT the inverse of the training transform.
    
    if obsm_key == "X_pca_scaled":
        # need to recompute scaler on the fly
        scaler = adata.uns['PC_scaler']
    elif not os.path.exists(os.path.join(args.model_dir, "scaler.npz")):
        scaler_path = os.path.join(args.model_dir, "scaler.npz")
        log.warning(f"scaler.npz missing at {scaler_path} — w2_raw will collapse to w2_scaled")
        scaler = None
    else:
        scaler_path = os.path.join(args.model_dir, "scaler.npz")
        scaler = dict(np.load(os.path.join(args.model_dir, "scaler.npz")))
        log.info(f"Scaler: loaded train-only stats from {scaler_path}")

    start_pre = np.asarray(start_adata.obsm[obsm_key][:, :n_dims])
    if scaler is not None:
        start_cells_fate = ((start_pre - scaler["mean"]) / scaler["std"]).astype(np.float32)
    else:
        start_cells_fate = start_pre.astype(np.float32)
    start_barcodes = np.asarray(start_adata.obs_names.astype(str))

    last_tp_idx = len(norm_tps) - 1
    x_ref = train_embs[last_tp_idx].astype(np.float32)
    y_ref = train_cts[last_tp_idx].astype(str)

    n_times = len(norm_tps)
    t_end_fate = float(n_times - 1)

    # ── Prepare W2 data (clone-filtered, scaled embeddings from test_cells.npz) ──
    # `clone_valid` mirrors pdp+'s `w2_clone_mask`: restricts both source and
    # target cells to clones in clone_proportions.index. compute_w2_per_clone
    # then evaluates the intersection set(src_clones) ∩ set(tgt_clones), which
    # is identical to pdp+'s `start_ad.obs.clones.unique()` after w2_clone_mask.
    test_ad = adata[adata.obs.Well == 2]
    bc_to_clone = dict(zip(test_ad.obs_names.astype(str),
                           test_ad.obs.clones.astype(str).values))
    clone_set = set(clone_proportions.index.astype(str))

    if test_barcodes is not None:
        test_clones_all = np.array(
            [bc_to_clone.get(bc, "__NONE__") for bc in test_barcodes.astype(str)]
        )
        clone_valid = np.isin(test_clones_all, list(clone_set))
    else:
        test_clones_all = np.array(["__NONE__"] * len(test_tps))
        clone_valid = np.zeros(len(test_tps), dtype=bool)

    tp1_mask = np.isclose(test_tps, norm_tps[1], atol=1e-3)
    tp_last_mask = np.isclose(test_tps, norm_tps[-1], atol=1e-3)

    src_mask_w2 = tp1_mask & clone_valid
    tgt_mask_w2 = tp_last_mask & clone_valid

    src_cells_w2 = test_emb[src_mask_w2].astype(np.float32)
    tgt_cells_w2 = test_emb[tgt_mask_w2].astype(np.float32)
    src_clones_w2 = test_clones_all[src_mask_w2]
    tgt_clones_w2 = test_clones_all[tgt_mask_w2]

    log.info(f"Fate start cells: {len(start_cells_fate)}")
    log.info(f"W2 src (tp_mid): {len(src_cells_w2)}, tgt (tp_last): {len(tgt_cells_w2)}")

    # ── Evaluate (ODE only — deterministic) ──
    mode = "ode"
    log.info(f"\n{'='*60}")
    log.info(f"Mode: {mode.upper()}")

    # Fate accuracy
    sim_fn_fate = fate_eval.make_otcfm_sim_fn(t_start=0.0, t_end=t_end_fate)

    fate_result = None
    if start_barcodes is not None:
        fate_result = fate_eval.run_fate_evaluation(
            start_cells=start_cells_fate,
            start_cell_ids=start_barcodes,
            model=model,
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
    else:
        log.warning("  No barcodes — skipping fate accuracy")

    # Population W2
    t_start_w2 = float(norm_tps[1])
    t_end_w2 = float(norm_tps[-1])
    sim_fn_w2 = fate_eval.make_otcfm_sim_fn(t_start=t_start_w2, t_end=t_end_w2)

    w2_result = fate_eval.compute_w2_population(
        src_cells=src_cells_w2,
        target_cells=tgt_cells_w2,
        model=model,
        sim_fn=sim_fn_w2,
        n_sims=1,
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
        model=model,
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
        "accuracy": fate_result["accuracy"] if fate_result else np.nan,
        "pearson_r": fate_result["pearson_r"] if fate_result else np.nan,
        "w2_scaled": w2_result["w2_scaled"],
        "w2_raw": w2_result["w2_raw"],
        "n_start_cells": fate_result["n_start_cells"] if fate_result else 0,
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
