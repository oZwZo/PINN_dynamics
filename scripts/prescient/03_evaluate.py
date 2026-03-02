"""
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

Usage
-----
python scripts/prescient/03_evaluate.py \
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
import argparse
import logging

import numpy as np
import pandas as pd
import torch
import scanpy as sc
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
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
# def compute_w2(x_sim: torch.Tensor, x_true: torch.Tensor) -> float:
#     """Exact W2 distance via torchcfm (POT backend)."""
#     try:
#         from torchcfm.optimal_transport import wasserstein
#         return float(wasserstein(x_sim.cpu(), x_true.cpu(), power=2, method="exact"))
#     except ImportError:
#         # Fallback: sliced Wasserstein via scipy if torchcfm unavailable
#         log.warning("torchcfm not available; falling back to sliced W2 (approximate)")
#         from scipy.stats import wasserstein_distance
#         x_s = x_sim.cpu().numpy()
#         x_t = x_true.cpu().numpy()
#         # mean over dimensions
#         return float(np.mean([wasserstein_distance(x_s[:, d], x_t[:, d])
#                                for d in range(x_s.shape[1])]))

def compute_w2(x_sim, x_true):
    """Compute exact W2 distance using POT library."""
    import ot
    x_s = x_sim.cpu().numpy().astype(np.float64)
    x_t = x_true.cpu().numpy().astype(np.float64)
    n_s = x_s.shape[0]
    n_t = x_t.shape[0]
    w_a = np.ones(n_s) / n_s
    w_b = np.ones(n_t) / n_t
    M = ot.dist(x_s, x_t, metric='sqeuclidean')
    w2_sq = ot.emd2(w_a, w_b, M)
    return float(np.sqrt(max(w2_sq, 0.0)))
def compute_accuracy(F_obs, F_hat):
    y_true, y_pred = F_obs.idxmax(1).tolist(), F_hat.idxmax(1).tolist()
    return sklearn.metrics.accuracy_score(y_true, y_pred)

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
    tp_first   = tp_values[1]
    tp_last    = tp_values[2]
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

    
    for rep in range(args.n_sims):
        x_i = torch.tensor(x_test_first).to(device)

        for _ in range(num_steps):
            z   = torch.randn(x_i.shape[0], x_i.shape[1], device=device) * config.train_sd
            x_i = net._step(x_i, dt=config.train_dt, z=z)

        w2 = compute_w2(x_i.detach(), x_true)
        w2_scores.append(w2)

        # fate bias on simulated final-timepoint cells
        fb = fate_bias_accuracy(
            x_sim       = x_i.detach().cpu().numpy(),
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


if __name__ == "__main__":
    main()
