"""
<<<<<<< HEAD
PRESCIENT Evaluation Script
=============================
Evaluates a trained PRESCIENT model on held-out test cells (Well == 2):
  1. W2 (Wasserstein-2) distance between simulated and true final-timepoint cells
  2. Fate bias accuracy: KNN-assigned fate fractions vs. observed test-cell fates

Model loading and simulation follow the PRESCIENT Python API:
  from prescient.train.model import SimpleNamespace, AutoGenerator

  config = SimpleNamespace(**torch.load(config_path))
  net    = AutoGenerator(config)
  # simulate manually:
  for _ in range(num_steps):
      z   = torch.randn_like(x_i) * config.train_sd
      x_i = net._step(x_i, dt=config.train_dt, z=z)
=======
PRESCIENT Combined Evaluation Script
=======================================
Evaluates a trained PRESCIENT model on held-out test cells (Well == 2).
Computes all metrics (fate accuracy, Pearson r, W2) in one run.

Outputs:
  - eval_combined.csv        (one row per sim_mode)
  - F_hat_{mode}.csv         (per-cell fate predictions)
  - w2_per_clone_{mode}.csv  (per-clone W2 distances)
>>>>>>> 2aa9d37e47e91f4f63e97a52f86f641c79576dd5

Usage
-----
python scripts/prescient/03_evaluate.py \
<<<<<<< HEAD
    --data_path  data/klein_subset.h5ad \
    --model_dir  logs/PRESCIENT/pca_run/PCA-softplus_1_500-1e-06/seed_2 \
    --obsm_key   X_pca --n_dims 30 \
    --output     logs/PRESCIENT/pca_run/eval_results.csv

python scripts/prescient/03_evaluate.py \
    --data_path  data/klein_subset.h5ad \
    --model_dir  logs/PRESCIENT/dm_run/DM-softplus_4_64-1e-06/seed_2 \
    --obsm_key   DM_EigenVectors --n_dims 10 \
    --output     logs/PRESCIENT/dm_run/eval_results.csv
"""

import os
=======
    --model_dir logs/PRESCIENT/pca_run/PCA-softplus_1_500-1e-06/seed_2 \
    --data_path data/klein_addpop.h5ad \
    --obsm_key X_pca --n_dims 30 --device cuda:0
"""

import os
import sys
>>>>>>> 2aa9d37e47e91f4f63e97a52f86f641c79576dd5
import argparse
import logging

import numpy as np
import pandas as pd
<<<<<<< HEAD
import torch
import scanpy as sc
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr
=======
import scanpy as sc
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import fate_eval_pipeline as fate_eval
>>>>>>> 2aa9d37e47e91f4f63e97a52f86f641c79576dd5

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
<<<<<<< HEAD
        description="Evaluate a trained PRESCIENT model (W2 + fate bias)"
    )
    p.add_argument("--data_path",     required=True, help="Path to original .h5ad file")
    p.add_argument("--model_dir",     required=True,
                   help="Path to trained model dir containing train.best.pt and config.pt")
    p.add_argument("--obsm_key",      default="X_pca",
                   choices=["X_pca", "DM_EigenVectors"],
                   help="obsm key used during training (default: X_pca)")
    p.add_argument("--n_dims",        type=int, default=30,
                   help="Embedding dimensions used during training (default: 30)")
    p.add_argument("--tp_col",        default="timepoint_tx_days",
                   help="Timepoint column in adata.obs (default: timepoint_tx_days)")
    p.add_argument("--well_col",      default="Well",
                   help="Well column for train/test split (default: Well)")
    p.add_argument("--celltype_col",  default="anno_man",
                   help="Cell-type annotation column (default: anno_man)")
    p.add_argument("--gpu",           default="cuda:0",
                   help="CUDA device string (default: cuda:0); 'cpu' to force CPU")
    p.add_argument("--n_sims",        type=int, default=10,
                   help="Number of simulation replicates for W2 estimation (default: 10)")
    p.add_argument("--output",        default=None,
                   help="Path to save evaluation CSV (default: model_dir/eval_results.csv)")
    return p.parse_args()


# ── Wasserstein-2 helper ──────────────────────────────────────────────────
def compute_w2(x_sim: torch.Tensor, x_true: torch.Tensor) -> float:
    """Exact W2 distance via torchcfm (POT backend)."""
    try:
        from torchcfm.optimal_transport import wasserstein
        return float(wasserstein(x_sim.cpu(), x_true.cpu(), power=2, method="exact"))
    except ImportError:
        # Fallback: sliced Wasserstein via scipy if torchcfm unavailable
        log.warning("torchcfm not available; falling back to sliced W2 (approximate)")
        from scipy.stats import wasserstein_distance
        x_s = x_sim.cpu().numpy()
        x_t = x_true.cpu().numpy()
        # mean over dimensions
        return float(np.mean([wasserstein_distance(x_s[:, d], x_t[:, d])
                               for d in range(x_s.shape[1])]))


# ── Fate bias helper ──────────────────────────────────────────────────────
def fate_bias_accuracy(
    x_sim: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test_final: np.ndarray,
    y_test_final: np.ndarray,
    k: int = 15,
) -> dict:
    """
    Assign simulated cells to cell types via KNN on train coords.
    Compare simulated fate fractions to observed test-cell fate fractions.

    Returns dict with:
        - pearson_r, pearson_p
        - simulated_fractions (dict: celltype → fraction)
        - observed_fractions  (dict: celltype → fraction)
        - cell_types          (list of cell type labels)
    """
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(x_train, y_train)

    sim_labels = knn.predict(x_sim)
    cell_types = sorted(np.unique(np.concatenate([y_train, y_test_final])))

    sim_frac = {ct: np.mean(sim_labels == ct) for ct in cell_types}
    obs_frac = {ct: np.mean(y_test_final == ct) for ct in cell_types}

    sim_vec = np.array([sim_frac[ct] for ct in cell_types])
    obs_vec = np.array([obs_frac[ct] for ct in cell_types])

    if sim_vec.std() < 1e-10 or obs_vec.std() < 1e-10:
        r, pval = 0.0, 1.0
    else:
        r, pval = pearsonr(sim_vec, obs_vec)

    return dict(
        pearson_r=r,
        pearson_p=pval,
        simulated_fractions=sim_frac,
        observed_fractions=obs_frac,
        cell_types=cell_types,
    )


def main():
    args = parse_args()

    device = args.gpu if (args.gpu != "cpu" and torch.cuda.is_available()) else "cpu"
    log.info(f"Using device: {device}")

    output_path = args.output or os.path.join(args.model_dir, "eval_results.csv")

    # ── load adata ────────────────────────────────────────────────────────
    log.info(f"Loading adata from {args.data_path}")
    adata = sc.read_h5ad(args.data_path)

    train_mask = adata.obs[args.well_col] != 2
    test_mask  = adata.obs[args.well_col] == 2
    adata_train = adata[train_mask]
    adata_test  = adata[test_mask]
    log.info(f"  Train: {train_mask.sum()}  Test: {test_mask.sum()}")

    tp_values = sorted(adata.obs[args.tp_col].unique())
    tp_first   = tp_values[0]
    tp_last    = tp_values[-1]
    log.info(f"  Timepoints: {tp_values}  →  simulate from t={tp_first} to t={tp_last}")

    # ── load PRESCIENT model ──────────────────────────────────────────────
    config_path = os.path.join(args.model_dir, "config.pt")
    train_pt    = os.path.join(args.model_dir, "train.best.pt")

    for p in (config_path, train_pt):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    from prescient.train.model import SimpleNamespace, AutoGenerator

    config = SimpleNamespace(**torch.load(config_path, weights_only=False))
    net    = AutoGenerator(config)
    checkpoint = torch.load(train_pt, map_location=device, weights_only=False)
    net.load_state_dict(checkpoint["model_state_dict"])
    net = net.to(device)
    net.eval()

    log.info(f"Model loaded: x_dim={config.x_dim}  layers={config.layers}  k_dim={config.k_dim}")

    # num_steps: simulate from start_t to train_t[-1]
    num_steps = int((config.train_t[-1] - config.start_t) / config.train_dt)
    log.info(f"  num_steps = {num_steps}  (train_dt={config.train_dt})")

    # ── prepare cell arrays ───────────────────────────────────────────────
    obsm_key = args.obsm_key
    n_dims   = args.n_dims

    # test cells at first and last timepoints
    test_first_mask = (adata_test.obs[args.tp_col] == tp_first).values
    test_last_mask  = (adata_test.obs[args.tp_col] == tp_last).values

    x_test_first = adata_test[test_first_mask].obsm[obsm_key][:, :n_dims].astype(np.float32)
    x_test_last  = adata_test[test_last_mask].obsm[obsm_key][:, :n_dims].astype(np.float32)
    y_test_last  = adata_test[test_last_mask].obs[args.celltype_col].values.astype(str)

    # train cells (for KNN fate bias)
    x_train = adata_train.obsm[obsm_key][:, :n_dims].astype(np.float32)
    y_train = adata_train.obs[args.celltype_col].values.astype(str)

    log.info(f"  Test cells at t={tp_first}: {x_test_first.shape[0]}")
    log.info(f"  Test cells at t={tp_last}:  {x_test_last.shape[0]}")

    x_true = torch.tensor(x_test_last)

    # ── W2 evaluation across replicates ──────────────────────────────────
    log.info(f"\nRunning {args.n_sims} simulation replicates for W2 ...")
    w2_scores = []
    fate_results = []

    with torch.no_grad():
        for rep in range(args.n_sims):
            x_i = torch.tensor(x_test_first).to(device)

            for _ in range(num_steps):
                z   = torch.randn(x_i.shape[0], x_i.shape[1], device=device) * config.train_sd
                x_i = net._step(x_i, dt=config.train_dt, z=z)

            w2 = compute_w2(x_i, x_true)
            w2_scores.append(w2)

            # fate bias on simulated final-timepoint cells
            fb = fate_bias_accuracy(
                x_sim       = x_i.cpu().numpy(),
                x_train     = x_train,
                y_train     = y_train,
                x_test_final= x_test_last,
                y_test_final= y_test_last,
            )
            fate_results.append(fb)

            log.info(f"  rep {rep+1:2d}/{args.n_sims}  W2={w2:.4f}  pearson_r={fb['pearson_r']:.4f}")

    w2_mean = np.mean(w2_scores)
    w2_std  = np.std(w2_scores)
    pr_mean = np.mean([f["pearson_r"] for f in fate_results])
    pr_std  = np.std( [f["pearson_r"] for f in fate_results])

    log.info(f"\n{'='*50}")
    log.info(f"W2 distance   : {w2_mean:.4f} ± {w2_std:.4f}")
    log.info(f"Fate Pearson r: {pr_mean:.4f} ± {pr_std:.4f}")
    log.info(f"{'='*50}")

    # ── save results ──────────────────────────────────────────────────────
    # Per-replicate summary
    summary_rows = []
    for rep_idx, (w2, fb) in enumerate(zip(w2_scores, fate_results)):
        summary_rows.append({
            "replicate":  rep_idx + 1,
            "w2":         w2,
            "pearson_r":  fb["pearson_r"],
            "pearson_p":  fb["pearson_p"],
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.loc["mean"] = df_summary.mean(numeric_only=True)
    df_summary.loc["std"]  = df_summary.std(numeric_only=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_summary.to_csv(output_path)
    log.info(f"\nResults saved → {output_path}")

    # Per-celltype fate fractions (from last replicate)
    last_fb = fate_results[-1]
    fate_rows = []
    for ct in last_fb["cell_types"]:
        fate_rows.append({
            "cell_type":           ct,
            "simulated_fraction":  last_fb["simulated_fractions"][ct],
            "observed_fraction":   last_fb["observed_fractions"][ct],
        })
    df_fate = pd.DataFrame(fate_rows)
    fate_csv = output_path.replace(".csv", "_fate_fractions.csv")
    df_fate.to_csv(fate_csv, index=False)
    log.info(f"Fate fractions → {fate_csv}")

    # Print fate table
    log.info(f"\nFate fractions (last replicate):")
    log.info(df_fate.to_string(index=False))
=======
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
                   help="Override config.train_sd at simulation time (no retraining). "
                        "Reduces SDE noise to inspect the W2-vs-fate trade-off — useful "
                        "when the model has high Pearson r / fate accuracy but inflated "
                        "W2 due to a noisy sim cloud. Default None keeps config.train_sd. "
                        "If set and --output_dir is not provided, results land in "
                        "<model_dir>/eval_sim_sd_<value>/ so the default eval is preserved.")
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
            f"Override sim noise: config.train_sd {config.train_sd} → {args.sim_sd} "
            f"(eval-time only; weights unchanged). fate_eval_pipeline.py reads "
            f"config.train_sd live each SDE step, so this takes effect immediately."
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
    tp_first = tp_values[0]   # day 2
    tp_mid = tp_values[1]     # day 4
    tp_last = tp_values[2]    # day 6
    log.info(f"Timepoints: {tp_values}")

    # Scaler convention: when training on a *_scaled obsm key, the data was already
    # z-scored upstream and the inverse lives in adata.uns. Pulling it from there
    # mirrors the SF2M/X_pca_scaled pattern. Raw obsm keys → no scaler (model lives
    # in raw space; w2_raw == w2_scaled).
    if args.obsm_key == "X_pca_scaled":
        scaler = adata.uns["PC_scaler"]
        log.info("Scaler: adata.uns['PC_scaler']")
    elif args.obsm_key == "DM_EigenVectors_scaled":
        scaler = adata.uns["DM_scaler"]
        log.info("Scaler: adata.uns['DM_scaler']")
    else:
        scaler = None
        if args.obsm_key == "DM_EigenVectors":
            log.warning(
                "obsm_key='DM_EigenVectors' (raw). Default train_sd=0.5 swamps "
                "DM raw std (≈0.0025), so this model likely failed to learn. "
                "Re-prepare with --obsm_key DM_EigenVectors_scaled and retrain."
            )
        else:
            log.info("Scaler: none → assuming model trained on raw obsm")

    # ── Prepare fate eval data ──
    # Start cells: cells at first timepoint that are in F_obs
    fobs_idx = F_obs.index.astype(str)
    valid_mask = adata.obs_names.astype(str).isin(fobs_idx)
    tp_start_mask = (adata.obs[args.tp_col] == tp_first).values
    start_mask = valid_mask & tp_start_mask

    # When obsm_key is *_scaled, adata.obsm[obsm_key] is already z-scored and matches
    # the model's input distribution; no forward-scaling needed. The scaler above is
    # used only by compute_w2_population to inverse endpoints back to raw units.
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
        sim_fn=sim_fn_fate,  # same SDE sim_fn works for W2
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
>>>>>>> 2aa9d37e47e91f4f63e97a52f86f641c79576dd5


if __name__ == "__main__":
    main()
