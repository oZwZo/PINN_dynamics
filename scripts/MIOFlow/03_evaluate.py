"""
MIOFlow Combined Evaluation Script
=====================================
Evaluates a trained MIOFlow model on held-out test cells (Well == 2).
Computes all metrics in one run: per-cell fate accuracy, Pearson r,
W2 (population + per-clone).

Uses MIOFlow's underlying Neural ODE directly (via torchdiffeq.odeint)
to simulate specific barcoded cells forward, NOT generate_points().
This gives true per-cell fate predictions comparable to other methods.

Cell extraction follows the pseudodynamics+ pattern (klein_eval_fate.py /
klein_eval_w2.py): cells are matched by barcode from adata.

Fate:  F_obs cells (tp2) → simulate t=2→6 → KNN classify → F_hat
W2:    Clone-filtered test cells (tp4) → simulate t=4→6 → compare to tp6

Outputs:
  - eval_combined.csv          (one row, sim_mode=ode)
  - F_hat_ode.csv              (per-cell fate predictions)
  - w2_per_clone_ode.csv       (per-clone W2)

Usage
-----
python scripts/MIOFlow/03_evaluate.py \
    --data_path data/klein_addpop.h5ad \
    --model_dir results/MIOFlow/klein_addpop_pca30 \
    --obsm_key X_pca --n_dims 30 --device cpu
"""

import os
import sys
import argparse
import logging

# ── tqdm monkey-patch (required before MIOFlow imports) ────────────────────
import tqdm as _tqdm
import tqdm.notebook
tqdm.notebook.tqdm = _tqdm.tqdm

import numpy as np
import pandas as pd
import torch
import scanpy as sc

from MIOFlow.models import make_model

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import fate_eval_pipeline as fate_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Combined evaluation for MIOFlow (fate accuracy + W2)"
    )
    p.add_argument("--data_path", required=True, help="Path to .h5ad file")
    p.add_argument("--model_dir", required=True,
                   help="Dir containing model_checkpoints.pt")
    p.add_argument("--F_obs_path", default="data/klein/F_obs.csv",
                   help="Ground-truth fate proportions CSV")
    p.add_argument("--clone_proportions", default="data/klein/clone_proportions.csv",
                   help="Clone proportions CSV")
    p.add_argument("--obsm_key", default="X_pca",
                   help="obsm key used during training")
    p.add_argument("--n_dims", type=int, default=30,
                   help="Embedding dimensions")
    p.add_argument("--tp_col", default="timepoint_tx_days")
    p.add_argument("--well_col", default="Well")
    p.add_argument("--celltype_col", default="Annotation",
                   help="Cell-type annotation column")
    p.add_argument("--k", type=int, default=15, help="KNN neighbors")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_dir", default=None,
                   help="Output directory (default: model_dir/)")
    return p.parse_args()


def main():
    args = parse_args()

    use_cuda = torch.cuda.is_available() and args.device != "cpu"
    device = "cuda" if use_cuda else "cpu"
    log.info(f"Device: {device}")

    output_dir = args.output_dir or args.model_dir
    os.makedirs(output_dir, exist_ok=True)

    obsm_key = args.obsm_key
    n_dims = args.n_dims

    # ── Load adata ──
    log.info(f"Loading adata: {args.data_path}")
    adata = sc.read_h5ad(args.data_path)

    tp_values = sorted(adata.obs[args.tp_col].unique())
    tp_first = tp_values[0]   # 2
    tp_mid = tp_values[1]     # 4
    tp_last = tp_values[-1]   # 6
    log.info(f"  Timepoints: {tp_values} (first={tp_first}, mid={tp_mid}, last={tp_last})")

    # ── Load model ──
    ckpt_path = os.path.join(args.model_dir, "model_checkpoints.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    opts = ckpt["options"]

    model = make_model(
        opts["model_features"], opts["layers"],
        activation=opts["activation"], scales=None, use_cuda=use_cuda,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info(f"Model loaded: features={opts['model_features']}  layers={opts['layers']}")

    # ── Load scaler (if exists) ──
    scaler_path = os.path.join(args.model_dir, "scaler.npz")
    scaler = dict(np.load(scaler_path)) if os.path.exists(scaler_path) else None
    log.info(f"Scaler: {'loaded' if scaler else 'None'}")

    # ── Load ground truth ──
    F_obs = pd.read_csv(args.F_obs_path, index_col=0)
    F_obs.index = F_obs.index.astype(str)
    F_obs = F_obs[F_obs.index.isin(adata.obs_names.astype(str))]
    clone_proportions = pd.read_csv(args.clone_proportions, index_col=0)

    # ═══════════════════════════════════════════════════════════════════════
    # FATE EVALUATION — extract start cells from adata by F_obs barcodes
    # Following klein_eval_fate.py pattern
    # ═══════════════════════════════════════════════════════════════════════
    fobs_idx_str = F_obs.index.astype(str)
    valid_mask = adata.obs_names.astype(str).isin(fobs_idx_str)
    start_cells_fate = np.array(
        adata[valid_mask].obsm[obsm_key][:, :n_dims]
    ).astype(np.float32)
    start_barcodes = list(adata[valid_mask].obs_names.astype(str))
    log.info(f"Fate start cells: {len(start_cells_fate)} (matched from F_obs)")

    # KNN reference: ALL train cells (all timepoints)
    adata_train = adata[adata.obs[args.well_col].astype(int) != 2]
    x_ref = np.array(adata_train.obsm[obsm_key][:, :n_dims]).astype(np.float32)
    y_ref = adata_train.obs[args.celltype_col].values.astype(str)
    log.info(f"KNN reference (all train cells): {len(x_ref)} cells")

    # Fate sim_fn: tp_first → tp_last (e.g. 2 → 6, 100 linspace steps)
    sim_fn_fate = fate_eval.make_mioflow_sim_fn(
        t_start=float(tp_first), t_end=float(tp_last)
    )

    fate_result = fate_eval.run_fate_evaluation(
        start_cells=start_cells_fate,
        start_cell_ids=start_barcodes,
        model=model,
        simulation_func=sim_fn_fate,
        F_obs=F_obs,
        x_ref=x_ref, y_ref=y_ref,
        n_sims=1,  # deterministic ODE
        k=args.k,
        device=device,
    )
    log.info(f"  Fate accuracy: {fate_result['accuracy']:.4f}")
    log.info(f"  Fate Pearson r: {fate_result['pearson_r']:.4f}")
    fate_result['F_hat'].to_csv(os.path.join(output_dir, "F_hat_ode.csv"))

    # ═══════════════════════════════════════════════════════════════════════
    # W2 EVALUATION — clone-filtered test cells from adata
    # Following klein_eval_w2.py pattern: tp4 → tp6
    # ═══════════════════════════════════════════════════════════════════════
    adata_test = adata[adata.obs[args.well_col].astype(int) == 2]
    w2_clone_mask = adata_test.obs.clones.isin(clone_proportions.index)
    w2_test_ad = adata_test[w2_clone_mask]

    src_ad_w2 = w2_test_ad[w2_test_ad.obs[args.tp_col] == tp_mid]
    tgt_ad_w2 = w2_test_ad[w2_test_ad.obs[args.tp_col] == tp_last]

    src_cells_w2 = np.array(src_ad_w2.obsm[obsm_key][:, :n_dims]).astype(np.float32)
    tgt_cells_w2 = np.array(tgt_ad_w2.obsm[obsm_key][:, :n_dims]).astype(np.float32)
    src_clones_w2 = src_ad_w2.obs.clones.values
    tgt_clones_w2 = tgt_ad_w2.obs.clones.values

    log.info(f"W2 src (tp{tp_mid}): {len(src_cells_w2)}, tgt (tp{tp_last}): {len(tgt_cells_w2)}")

    # W2 sim_fn: tp_mid → tp_last (e.g. 4 → 6, 100 linspace steps)
    sim_fn_w2 = fate_eval.make_mioflow_sim_fn(
        t_start=float(tp_mid), t_end=float(tp_last)
    )

    # Population W2
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
    df_clone_w2.to_csv(os.path.join(output_dir, "w2_per_clone_ode.csv"), index=False)
    log.info(f"  Per-clone W2: {len(df_clone_w2)} clones")

    # ═══════════════════════════════════════════════════════════════════════
    # SAVE eval_combined.csv
    # ═══════════════════════════════════════════════════════════════════════
    mode = "ode"
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
