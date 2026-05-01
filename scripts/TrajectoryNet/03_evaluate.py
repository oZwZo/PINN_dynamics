"""
TrajectoryNet Combined Evaluation Script
===========================================
Evaluates a trained TrajectoryNet model on held-out test cells.
Computes all metrics (fate accuracy, Pearson r, W2) in one run.

Outputs:
  - eval_combined.csv        (one row per sim_mode)
  - F_hat_{mode}.csv         (per-cell fate predictions)
  - w2_per_clone_{mode}.csv  (per-clone W2 distances)

Usage
-----
python scripts/TrajectoryNet/03_evaluate.py \
    --data_dir logs/TrajectoryNet/pca30 \
    --model_dir logs/TrajectoryNet/pca30/model \
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

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import fate_eval_pipeline as fate_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Combined evaluation for TrajectoryNet (fate accuracy + W2)"
    )
    p.add_argument("--data_dir", required=True,
                   help="Dir with scaler.npz, test_cells.npz, train_meta.npz, config.json")
    p.add_argument("--model_dir", required=True,
                   help="Dir with checkpt.pth and train_args.json")
    p.add_argument("--data_path", default="data/klein/klein_addpop.h5ad",
                   help="Path to .h5ad file (for barcodes, clones)")
    p.add_argument("--F_obs_path", default="data/klein/F_obs.csv",
                   help="Ground-truth fate proportions CSV")
    p.add_argument("--clone_proportions", default="data/klein/clone_proportions.csv",
                   help="Clone proportions CSV")
    p.add_argument("--n_sims_fate", type=int, default=100,
                   help="Trajectories per cell for fate (deterministic, so 1 used)")
    p.add_argument("--n_sims_w2", type=int, default=10,
                   help="Trajectories per cell for W2 (deterministic, so 1 used)")
    p.add_argument("--k", type=int, default=15, help="KNN neighbors")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default=None,
                   help="Output directory (default: data_dir/)")
    return p.parse_args()


def load_trajectorynet_model(model_dir, data_path, device):
    """Load a trained TrajectoryNet model from checkpoint + train_args.json."""
    from TrajectoryNet.parse import parser as tjn_parser
    from TrajectoryNet.train_misc import (
        build_model_tabular,
        create_regularization_fns,
    )
    from TrajectoryNet import dataset

    train_args_path = os.path.join(model_dir, "train_args.json")
    with open(train_args_path) as f:
        train_args = json.load(f)

    parser_args = [
        "--dataset", data_path,
        "--embedding_name", train_args["embedding_name"],
        "--max_dim", str(train_args["max_dim"]),
        "--dims", train_args["dims"],
        "--time_scale", str(train_args["time_scale"]),
        "--layer_type", train_args["layer_type"],
        "--nonlinearity", train_args["nonlinearity"],
        "--save", os.path.abspath(model_dir),
        "--gpu", str(device.index if device.type == "cuda" else 0),
    ]
    tjn_args = tjn_parser.parse_args(parser_args)

    data = dataset.SCData.factory(tjn_args.dataset, tjn_args)
    tjn_args.data = data
    tjn_args.timepoints = data.get_unique_times()
    tjn_args.int_tps = (np.arange(max(tjn_args.timepoints) + 1) + 1.0) * tjn_args.time_scale

    log.info(f"  int_tps: {tjn_args.int_tps}")

    regularization_fns, _ = create_regularization_fns(tjn_args)
    model = build_model_tabular(tjn_args, data.get_shape()[0], regularization_fns)

    ckpt_path = os.path.join(model_dir, "checkpt.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    log.info(f"  Loaded model from {ckpt_path}")

    return model, tjn_args


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    log.info(f"Device: {device}")

    output_dir = args.output_dir or args.data_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── Load config ──
    config_path = os.path.join(args.data_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    in_out_dim = config["in_out_dim"]
    norm_tps = config["normalized_timepoints"]
    log.info(f"Config: in_out_dim={in_out_dim}, norm_tps={norm_tps}")

    # ── Load model ──
    data_npz_path = os.path.join(args.data_dir, "klein_train.npz")
    model, tjn_args = load_trajectorynet_model(args.model_dir, data_npz_path, device)
    int_tps = tjn_args.int_tps

    # ── Load scaler ──
    scaler_path = os.path.join(args.data_dir, "scaler.npz")
    scaler = dict(np.load(scaler_path)) if os.path.exists(scaler_path) else None
    log.info(f"Scaler: {'loaded' if scaler else 'None'}")

    def transform(x):
        normed_x = (x - scaler["mean"])/scaler["std"]
        return normed_x
    def inverse_transform(normed_x):
        x =  scaler["mean"] + normed_x * scaler["std"]
        return x

    # ── Load npz data ──
    train_meta = np.load(os.path.join(args.data_dir, "train_meta.npz"), allow_pickle=True)
    train_embs = train_meta["embeddings"]
    train_cts = train_meta["celltypes"]

    test_data = np.load(os.path.join(args.data_dir, "test_cells.npz"), allow_pickle=True)
    test_emb = test_data["embeddings"]
    test_tps = test_data["timepoints"]
    test_barcodes = test_data["barcodes"] if "barcodes" in test_data else None

    # ── Load adata for clones ──
    log.info(f"Loading adata: {args.data_path}")
    adata = sc.read_h5ad(args.data_path)
    clone_proportions = pd.read_csv(args.clone_proportions, index_col=0)

    F_obs = pd.read_csv(args.F_obs_path, index_col=0)
    F_obs.index = F_obs.index.astype(str)
    F_obs = F_obs[F_obs.index.isin(adata.obs_names.astype(str))]

    # ── Prepare fate eval data ──
    # Match PRESCIENT: start = adata ∩ F_obs.index ∩ tp == tp_first
    n_dims = in_out_dim
    tp_values = sorted(adata.obs.timepoint_tx_days.unique())
    tp_first = tp_values[0]
    fobs_idx = F_obs.index.astype(str)
    valid_mask = adata.obs_names.astype(str).isin(fobs_idx)
    tp_start_mask = (adata.obs.timepoint_tx_days == tp_first).values
    start_mask = valid_mask & tp_start_mask
    start_adata_fate = adata[start_mask]
    fate_start_x = start_adata_fate.obsm[config['obsm_key']][:, :n_dims].astype(np.float32)
    start_cells_fate = transform(fate_start_x)
    start_barcodes = list(start_adata_fate.obs_names.astype(str))

    # KNN reference: all train cells across all timepoints (matches PRESCIENT)
    x_ref = np.concatenate(
        [train_embs[i].astype(np.float32) for i in range(len(train_embs))], axis=0
    )
    y_ref = np.concatenate(
        [train_cts[i].astype(str) for i in range(len(train_cts))], axis=0
    )

    # Fate: tp0 → tp_last, using integration time indices
    int_tp_start_fate = 0.0
    int_tp_end_fate = 2.0

    # ── Prepare W2 data ──
    test_ad = adata[adata.obs.Well == 2]
    # W2 integration times
    int_tp_start_w2 = 1.0
    int_tp_end_w2 = 2.0

    # Clone-filtered W2 from adata
    n_dims = in_out_dim
    w2_clone_mask = test_ad.obs.clones.isin(clone_proportions.index)
    w2_test_ad = test_ad[w2_clone_mask]
    src_ad_w2 = w2_test_ad[w2_test_ad.obs.timepoint_tx_days == 4]
    tgt_ad_w2 = w2_test_ad[w2_test_ad.obs.timepoint_tx_days == 6]

    obsm_key = config.get("obsm_key", "X_pca" if n_dims == 30 else "DM_EigenVectors")
    src_cells_w2_clone = src_ad_w2.obsm[obsm_key][:, :n_dims].astype(np.float32)
    src_cells_w2_clone = transform(src_cells_w2_clone) # standardize
    
    tgt_cells_w2_clone = tgt_ad_w2.obsm[obsm_key][:, :n_dims].astype(np.float32)
    tgt_cells_w2_clone = transform(tgt_cells_w2_clone) # standardize

    src_clones_w2 = src_ad_w2.obs.clones.values
    tgt_clones_w2 = tgt_ad_w2.obs.clones.values

    log.info(f"Fate start cells: {len(start_cells_fate)}")
    log.info(f"W2 src (tp1): {len(src_cells_w2_clone)}, tgt (tp_last): {len(tgt_cells_w2_clone)}")

    # ── Evaluate (ODE only — deterministic CNF) ──
    mode = "ode"
    log.info(f"\n{'='*60}")
    log.info(f"Mode: {mode.upper()}")

    # Fate accuracy
    sim_fn_fate = fate_eval.make_trajectorynet_sim_fn(
        int_tp_start=int_tp_start_fate, int_tp_end=int_tp_end_fate)

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
            device=str(device),
        )
        log.info(f"  Fate accuracy: {fate_result['accuracy']:.4f}")
        log.info(f"  Fate Pearson r: {fate_result['pearson_r']:.4f}")
        fate_result['F_hat'].to_csv(os.path.join(args.model_dir, f"F_hat_{mode}.csv"))
    else:
        log.warning("  No barcodes in test_cells.npz — skipping fate accuracy")

    # Population W2
    sim_fn_w2 = fate_eval.make_trajectorynet_sim_fn(
        int_tp_start=int_tp_start_w2, int_tp_end=int_tp_end_w2)

    w2_result = fate_eval.compute_w2_population(
        src_cells=src_cells_w2_clone,     # standardized source emb
        target_cells=tgt_cells_w2_clone,  # standardized target emb
        model=model,
        sim_fn=sim_fn_w2,
        n_sims=1,
        device=str(device),
        scaler=scaler,
    )
    log.info(f"  W2 scaled: {w2_result['w2_scaled']:.4f}")
    log.info(f"  W2 raw:    {w2_result['w2_raw']:.4f}")

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
    combined_path = os.path.join(args.model_dir, "eval_combined.csv")
    df_combined.to_csv(combined_path, index=False)
    log.info(f"\n{'='*60}")
    log.info(f"Combined results → {combined_path}")
    log.info(f"\n{df_combined.to_string(index=False)}")
    log.info("Done.")

    # Per-clone W2
    df_clone_w2 = fate_eval.compute_w2_per_clone(
        src_cells=src_cells_w2_clone,
        src_clone_ids=src_clones_w2,
        target_cells=tgt_cells_w2_clone,
        target_clone_ids=tgt_clones_w2,
        model=model,
        sim_fn=sim_fn_w2,
        n_sims=1,
        device=str(device),
        scaler=scaler,
    )
    df_clone_w2.to_csv(os.path.join(args.model_dir, f"w2_per_clone_{mode}.csv"), index=False)
    log.info(f"  Per-clone W2: {len(df_clone_w2)} clones")


if __name__ == "__main__":
    main()
