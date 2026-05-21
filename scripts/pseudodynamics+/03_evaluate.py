"""
pseudodynamics+ Combined Evaluation Script
=============================================
Evaluates a trained pseudodynamics+ model on held-out test cells.
Computes all metrics (fate accuracy, Pearson r, W2) in one run across
all simulation modes (ode, sde, sb).

Outputs:
  - eval_combined.csv        (one row per sim_mode)
  - F_hat_{mode}.csv         (per-cell fate predictions)
  - w2_per_clone_{mode}.csv  (per-clone W2 distances)

Usage
-----
python scripts/pseudodynamics+/03_evaluate.py \
    --config_path logs/klein_DM_10_.../pde_params_tsense/V0_config.json \
    --data_path data/klein/klein_addpop.h5ad \
    --device cuda:0
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
        description="Combined evaluation for pseudodynamics+ (fate accuracy + W2)"
    )
    p.add_argument("--config_path", required=True,
                   help="pdp training config JSON")
    p.add_argument("--data_path", default="data/klein/klein_addpop.h5ad",
                   help="Path to .h5ad file (for barcodes, clones, obsm)")
    p.add_argument("--F_obs_path", default="data/klein/F_obs.csv",
                   help="Ground-truth fate proportions CSV")
    p.add_argument("--clone_proportions", default="data/klein/clone_proportions.csv",
                   help="Clone proportions CSV for W2 clone filtering")
    p.add_argument("--celltype_col", default="Annotation")
    p.add_argument("--sim_modes", nargs="+", default=["ode", "sde", "sb"],
                   choices=["ode", "sde", "sb"],
                   help="Simulation modes to evaluate (default: all three)")
    p.add_argument("--t_end_fate", type=float, default=4.0,
                   help="Normalised end time for fate eval: 0 -> t_end_fate")
    p.add_argument("--t_end_w2", type=float, default=4.0,
                   help="Normalised end time for W2 eval: 2 -> t_end_w2")
    p.add_argument("--n_steps", type=int, default=200,
                   help="Euler-Maruyama discretisation steps (SDE/SB)")
    p.add_argument("--noise_scale", type=float, default=1.0,
                   help="Stochastic noise multiplier (SDE/SB)")
    p.add_argument("--n_sims_fate", type=int, default=100,
                   help="Trajectories per cell for fate accuracy")
    p.add_argument("--n_sims_w2", type=int, default=10,
                   help="Trajectories per cell for W2")
    p.add_argument("--k", type=int, default=15,
                   help="KNN neighbors for fate assignment")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default=None,
                   help="Output directory (default: derived from config)")
    return p.parse_args()


def build_sim_fn(mode, t_start, t_end, n_steps, noise_scale):
    """Build simulation function for a given mode."""
    if mode == "ode":
        return fate_eval.make_pseudodynamics_sim_fn(
            t_start_norm=t_start, t_end_norm=t_end)
    elif mode == "sde":
        return fate_eval.make_pseudodynamics_sde_sim_fn(
            t_start_norm=t_start, t_end_norm=t_end,
            n_steps=n_steps, noise_scale=noise_scale)
    elif mode == "sb":
        return fate_eval.make_pseudodynamics_sb_sim_fn(
            t_start_norm=t_start, t_end_norm=t_end,
            n_steps=n_steps, noise_scale=noise_scale)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    args = parse_args()
    import pseudodynamics as pdp

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    log.info(f"Device: {device}")

    # ── Load config & model ──
    config = pdp.ExperimentConfig(args.config_path)
    n_dims = config.dataset_config['n_dimension']
    cellstate_key = config.dataset_config.get("cellstate_key", None)

    pde_model = pdp.models.pde_params.load_from_checkpoint(
        config.find_lastest_ckpt()
    ).to(device)
    pde_model.eval()
    log.info(f"Model loaded: cellstate_key={cellstate_key}, n_dims={n_dims}")

    # ── Output dir ──
    if args.output_dir is None:
        name = config.experiment_config['save_dir'].split("/")[-2]
        output_dir = f"results/pseudodynamics+/{name}"
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── Load data ──
    log.info(f"Loading adata: {args.data_path}")
    adata = sc.read_h5ad(args.data_path)

    F_obs = pd.read_csv(args.F_obs_path, index_col=0)
    F_obs.index = F_obs.index.astype(str)
    F_obs = F_obs[F_obs.index.isin(adata.obs_names.astype(str))]

    clone_proportions = pd.read_csv(args.clone_proportions, index_col=0)

    # ── Compute scaler ──
    scaler = fate_eval.compute_scaler_from_adata(adata, cellstate_key, n_dims)
    log.info(f"Scaler: {'computed from adata' if scaler else 'None (no _scaled suffix)'}")

    # ── Prepare fate eval data ──
    # Start cells: cells in F_obs (at earliest timepoint, all wells for fate)
    fobs_idx = F_obs.index.astype(str)
    valid_mask = adata.obs_names.astype(str).isin(fobs_idx)
    start_cells_fate = adata[valid_mask].obsm[cellstate_key][:, :n_dims].astype(np.float32)
    start_cell_ids = list(adata[valid_mask].obs_names.astype(str))

    # KNN reference: training cells
    adata_train = adata[adata.obs.Well != 2]
    x_ref = adata_train.obsm[cellstate_key][:, :n_dims].astype(np.float32)
    y_ref = adata_train.obs[args.celltype_col].values.astype(str)

    # ── Prepare W2 eval data ──
    test_ad = adata[adata.obs.Well == 2]
    w2_clone_mask = test_ad.obs.clones.isin(clone_proportions.index)
    w2_test_ad = test_ad[w2_clone_mask].copy()

    src_ad_w2 = w2_test_ad[w2_test_ad.obs.timepoint_tx_days == 4]
    tgt_ad_w2 = w2_test_ad[w2_test_ad.obs.timepoint_tx_days == 6]

    src_cells_w2 = src_ad_w2.obsm[cellstate_key][:, :n_dims].astype(np.float32)
    tgt_cells_w2 = tgt_ad_w2.obsm[cellstate_key][:, :n_dims].astype(np.float32)
    src_clones_w2 = src_ad_w2.obs.clones.values
    tgt_clones_w2 = tgt_ad_w2.obs.clones.values

    log.info(f"Fate start cells: {len(start_cells_fate)}")
    log.info(f"W2 src cells (t4): {len(src_cells_w2)}, tgt cells (t6): {len(tgt_cells_w2)}")

    # ── Evaluate each mode ──
    combined_rows = []

    for mode in args.sim_modes:
        log.info(f"\n{'='*60}")
        log.info(f"Mode: {mode.upper()}")

        n_sims_fate = 1 if mode == "ode" else args.n_sims_fate
        n_sims_w2 = 1 if mode == "ode" else args.n_sims_w2

        # ── Fate accuracy ──
        sim_fn_fate = build_sim_fn(mode, 0.0, args.t_end_fate,
                                   args.n_steps, args.noise_scale)
        fate_result = fate_eval.run_fate_evaluation(
            start_cells=start_cells_fate,
            start_cell_ids=start_cell_ids,
            model=pde_model,
            simulation_func=sim_fn_fate,
            F_obs=F_obs,
            x_ref=x_ref, y_ref=y_ref,
            n_sims=n_sims_fate,
            k=args.k,
            device=device,
        )
        log.info(f"  Fate accuracy: {fate_result['accuracy']:.4f}")
        log.info(f"  Fate Pearson r: {fate_result['pearson_r']:.4f}")

        # Save F_hat
        fate_result['F_hat'].to_csv(os.path.join(output_dir, f"F_hat_{mode}.csv"))

        # ── Population W2 ──
        sim_fn_w2 = build_sim_fn(mode, 2.0, args.t_end_w2,
                                 args.n_steps, args.noise_scale)
        w2_result = fate_eval.compute_w2_population(
            src_cells=src_cells_w2,
            target_cells=tgt_cells_w2,
            model=pde_model,
            sim_fn=sim_fn_w2,
            n_sims=n_sims_w2,
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
            model=pde_model,
            sim_fn=sim_fn_w2,
            n_sims=n_sims_w2,
            device=device,
            scaler=scaler,
        )
        df_clone_w2.to_csv(os.path.join(output_dir, f"w2_per_clone_{mode}.csv"), index=False)
        log.info(f"  Per-clone W2: {len(df_clone_w2)} clones, "
                 f"mean_scaled={df_clone_w2['w2_scaled'].mean():.4f}")

        combined_rows.append({
            "sim_mode": mode,
            "accuracy": fate_result["accuracy"],
            "pearson_r": fate_result["pearson_r"],
            "w2_scaled": w2_result["w2_scaled"],
            "w2_raw": w2_result["w2_raw"],
            "n_start_cells": fate_result["n_start_cells"],
            "n_sims_fate": n_sims_fate,
            "k": args.k,
        })

    # ── Save combined CSV ──
    df_combined = pd.DataFrame(combined_rows)
    combined_path = os.path.join(output_dir, "eval_combined.csv")
    df_combined.to_csv(combined_path, index=False)
    log.info(f"\n{'='*60}")
    log.info(f"Combined results → {combined_path}")
    log.info(f"\n{df_combined.to_string(index=False)}")
    log.info("Done.")


if __name__ == "__main__":
    main()
