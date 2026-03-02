"""
eval_fate_acc.py — Per-cell fate accuracy evaluation for PRESCIENT
====================================================================

Implements the TIGON-style fate accuracy metric (Yeo et al., NatComms 2021;
scdiffeq-analyses figure_2/fate_prediction/TIGON/run_tigon.ipynb):

  1.  Start cells  : filter adata to cells in F_obs.index (cells with
                     clone-traced fate labels).
  2.  Simulation   : for each start cell, replicate it n_sims times and
                     integrate forward with the PRESCIENT SDE.
  3.  KNN labelling: assign each simulated endpoint a cell-type label via
                     k-nearest-neighbours on training data.
  4.  F_hat        : for each start cell, count the fraction of n_sims
                     endpoints assigned to each cell type.
  5.  Accuracy     : sklearn accuracy_score(F_obs.idxmax(1), F_hat.idxmax(1))
  6.  Pearson r    : correlation between per-cell-type mean proportions in
                     F_obs and F_hat (population-level distribution match).

Public API
----------
  predict_fates_per_cell(start_cells, net, config, knn, cell_types, n_sims, device)
      → F_hat  (np.ndarray, shape n_cells × n_cell_types)

  compute_fate_accuracy(F_obs_df, F_hat_df)
      → dict(accuracy, pearson_r, pearson_p, y_true, y_pred)

  evaluate_fate_accuracy(adata, net, config, F_obs, obsm_key, n_dims,
                         celltype_col, tp_col, tp_start, train_mask,
                         n_sims, k, device)
      → dict(accuracy, pearson_r, pearson_p, F_hat, F_obs_aligned,
             n_start_cells, y_true, y_pred)

Standalone CLI
--------------
  python eval_fate_acc.py \
      --data_path   data/klein_subset.h5ad \
      --model_dir   logs/PRESCIENT/pca_run/PCA-softplus_1_500-1e-06/seed_2 \
      --F_obs_path  data/klein/F_obs.csv \
      --obsm_key    X_pca --n_dims 30 \
      --tp_start    2 \
      --output      logs/PRESCIENT/pca_run/fate_acc_results.csv
"""

import logging
import numpy as np
import pandas as pd
import sklearn.metrics
from scipy.stats import pearsonr
from sklearn.neighbors import KNeighborsClassifier
import torch

log = logging.getLogger(__name__)


# ─── Core simulation + labelling ─────────────────────────────────────────────

def predict_fates_per_cell(
    start_cells: np.ndarray,
    net,
    config,
    knn: KNeighborsClassifier,
    cell_types: list,
    n_sims: int = 100,
    device: str = "cpu",
    log_progress: bool = True,
) -> np.ndarray:
    """
    For every start cell, simulate n_sims stochastic trajectories forward and
    assign each endpoint a cell-type label via the pre-fitted KNN.

    Parameters
    ----------
    start_cells : ndarray, shape (n_cells, n_dims)
        Embedding coordinates of cells with known fates (float32).
    net         : PRESCIENT AutoGenerator (eval mode, on device).
    config      : PRESCIENT SimpleNamespace with train_t, start_t, train_dt, train_sd.
    knn         : Fitted KNeighborsClassifier mapping embedding → cell type.
    cell_types  : Ordered list of cell-type strings (columns of F_hat).
    n_sims      : Number of simulated trajectories per start cell.
    device      : Torch device string.
    log_progress: Print progress every 50 cells.

    Returns
    -------
    F_hat_arr : ndarray, shape (n_cells, n_cell_types)
        Predicted fate probability per cell (each row sums to 1).
    """
    num_steps = int((config.train_t[-1] - config.start_t) / config.train_dt)
    n_cells   = len(start_cells)
    ct_index  = {ct: i for i, ct in enumerate(cell_types)}
    F_hat_arr = np.zeros((n_cells, len(cell_types)), dtype=np.float32)

    net.eval()
    
    for i in range(n_cells):
        if log_progress and i % 50 == 0:
            log.info(f"  Fate sim: cell {i}/{n_cells}")

        # Replicate single start cell n_sims times → (n_sims, n_dims)
        x_i = torch.tensor(
            np.tile(start_cells[i], (n_sims, 1)), dtype=torch.float32
        ).to(device)

        for _ in range(num_steps):
            z   = torch.randn_like(x_i) * config.train_sd
            x_i = net._step(x_i, dt=config.train_dt, z=z)

        endpoints = x_i.detach().cpu().numpy()          # (n_sims, n_dims)
        labels    = knn.predict(endpoints)     # (n_sims,)

        for ct, frac in zip(
            *np.unique(labels, return_counts=True)
        ):
            if ct in ct_index:
                F_hat_arr[i, ct_index[ct]] = frac / n_sims

    return F_hat_arr


# ─── Accuracy metric ──────────────────────────────────────────────────────────

def compute_fate_accuracy(
    F_obs_df: pd.DataFrame,
    F_hat_df: pd.DataFrame,
) -> dict:
    """
    Compare per-cell predicted vs. ground-truth fate distributions.

    Parameters
    ----------
    F_obs_df : DataFrame, shape (n_cells, n_cell_types)
        Ground-truth fate probabilities from clone tracing.
    F_hat_df : DataFrame, shape (n_cells, n_cell_types)
        Predicted fate probabilities from model simulation.
        Must have the same index as F_obs_df.

    Returns
    -------
    dict with keys:
        accuracy   : float — fraction of cells where argmax fate matches
        pearson_r  : float — Pearson r between mean per-celltype proportions
        pearson_p  : float — corresponding p-value
        y_true     : list[str]  — dominant fate per cell (ground truth)
        y_pred     : list[str]  — dominant fate per cell (predicted)
    """
    shared_cols = sorted(set(F_obs_df.columns) & set(F_hat_df.columns))
    if not shared_cols:
        raise ValueError("F_obs and F_hat share no cell-type columns.")

    obs = F_obs_df[shared_cols]
    hat = F_hat_df[shared_cols]

    y_true = obs.idxmax(axis=1).tolist()
    y_pred = hat.idxmax(axis=1).tolist()

    # Cells with zero total prediction → "Undifferentiated"
    zero_mask = hat.sum(axis=1) == 0
    for idx in np.where(zero_mask.values)[0]:
        y_pred[idx] = "Undifferentiated"

    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)

    # Population-level Pearson r
    obs_vec = obs.mean(axis=0).values
    hat_vec = hat.mean(axis=0).values
    if obs_vec.std() < 1e-10 or hat_vec.std() < 1e-10:
        r, p = 0.0, 1.0
    else:
        r, p = pearsonr(obs_vec, hat_vec)

    return dict(accuracy=accuracy, pearson_r=r, pearson_p=p,
                y_true=y_true, y_pred=y_pred)


# ─── End-to-end evaluation ────────────────────────────────────────────────────

def evaluate_fate_accuracy(
    adata,
    net,
    config,
    F_obs: pd.DataFrame,
    obsm_key: str,
    n_dims: int,
    celltype_col: str,
    tp_col: str,
    tp_start,
    train_mask,
    n_sims: int = 100,
    k: int = 20,
    device: str = "cpu",
) -> dict:
    """
    End-to-end per-cell fate accuracy evaluation.

    Parameters
    ----------
    adata        : AnnData — full dataset (train + test cells).
    net          : trained PRESCIENT AutoGenerator.
    config       : PRESCIENT config SimpleNamespace.
    F_obs        : DataFrame, index = cell barcodes, columns = cell types,
                   values = ground-truth fate probabilities (clone tracing).
    obsm_key     : Embedding key used during training ('X_pca' or 'DM_EigenVectors').
    n_dims       : Number of embedding dimensions used during training.
    celltype_col : adata.obs column with cell-type labels (for KNN reference).
    tp_col       : adata.obs column with timepoint values.
    tp_start     : Timepoint value of start cells (must match F_obs cells).
    train_mask   : Boolean array selecting training cells (for KNN reference).
    n_sims       : Trajectories per start cell (default 100).
    k            : KNN neighbours for cell-type assignment (default 20).
    device       : Torch device string.

    Returns
    -------
    dict with keys:
        accuracy        : float
        pearson_r       : float
        pearson_p       : float
        n_start_cells   : int
        F_hat           : DataFrame (n_cells × n_cell_types)
        F_obs_aligned   : DataFrame (n_cells × n_cell_types, F_obs subset)
        y_true          : list[str]
        y_pred          : list[str]
    """
    # ── 1. Select start cells ──────────────────────────────────────────────
    # Match F_obs index (string) against adata.obs_names (string)
    obs_names_str = adata.obs_names.astype(str)
    fobs_idx_str  = F_obs.index.astype(str)

    valid_mask  = obs_names_str.isin(fobs_idx_str)
    tp_mask     = (adata.obs[tp_col].astype(str) == str(tp_start)).values
    start_mask  = valid_mask & tp_mask

    if start_mask.sum() == 0:
        raise ValueError(
            f"No cells found matching F_obs.index at tp_start={tp_start}.\n"
            f"  F_obs has {len(F_obs)} entries; "
            f"adata has {adata.n_obs} cells.\n"
            f"  Sample F_obs index: {list(fobs_idx_str[:5])}\n"
            f"  Sample adata obs_names: {list(obs_names_str[:5])}"
        )

    adata_start  = adata[start_mask]
    # Re-index F_obs to match adata_start order
    F_obs_start  = F_obs.loc[adata_start.obs_names.astype(str)]
    F_obs_start.index = adata_start.obs_names

    start_cells = adata_start.obsm[obsm_key][:, :n_dims].astype(np.float32)
    log.info(f"Start cells with known fate at tp={tp_start}: {len(start_cells)}")

    # ── 2. Build KNN on training cells ────────────────────────────────────
    adata_train = adata[train_mask]
    x_ref = adata_train.obsm[obsm_key][:, :n_dims].astype(np.float32)
    y_ref = adata_train.obs[celltype_col].values.astype(str)

    cell_types = sorted(F_obs.columns.tolist())
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(x_ref, y_ref)
    log.info(f"KNN fitted on {len(x_ref)} training cells, k={k}")

    # ── 3. Simulate and build F_hat ────────────────────────────────────────
    log.info(f"Simulating {n_sims} trajectories per cell …")
    F_hat_arr = predict_fates_per_cell(
        start_cells=start_cells,
        net=net,
        config=config,
        knn=knn,
        cell_types=cell_types,
        n_sims=n_sims,
        device=device,
    )
    F_hat = pd.DataFrame(F_hat_arr, index=adata_start.obs_names, columns=cell_types)

    # ── 4. Compute accuracy ────────────────────────────────────────────────
    result = compute_fate_accuracy(F_obs_start, F_hat)
    result.update(
        n_start_cells=len(start_cells),
        F_hat=F_hat,
        F_obs_aligned=F_obs_start,
    )

    log.info(
        f"Fate accuracy : {result['accuracy']:.4f}  "
        f"Pearson r : {result['pearson_r']:.4f}"
    )
    return result


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Per-cell fate accuracy evaluation for a trained PRESCIENT model"
    )
    p.add_argument("--data_path",    required=True,
                   help="Path to .h5ad file")
    p.add_argument("--model_dir",    required=True,
                   help="PRESCIENT model dir with train.best.pt and config.pt")
    p.add_argument("--F_obs_path",   required=True,
                   help="Path to F_obs.csv (cells × cell-types, index = barcodes)")
    p.add_argument("--obsm_key",     default="X_pca",
                   choices=["X_pca", "DM_EigenVectors"])
    p.add_argument("--n_dims",       type=int, default=30)
    p.add_argument("--tp_col",       default="timepoint_tx_days")
    p.add_argument("--well_col",     default="Well")
    p.add_argument("--celltype_col", default="anno_man")
    p.add_argument("--tp_start",     default=None,
                   help="Timepoint value of start cells. "
                        "Default: first timepoint in adata.")
    p.add_argument("--n_sims",       type=int, default=100,
                   help="Simulated trajectories per start cell (default 100)")
    p.add_argument("--k",            type=int, default=20,
                   help="KNN neighbours for cell-type assignment (default 20)")
    p.add_argument("--gpu",          default="cuda:0")
    p.add_argument("--output",       default=None,
                   help="Output CSV path (default: model_dir/fate_acc_results.csv)")
    return p.parse_args()


def main():
    import os, scanpy as sc
    from prescient.train.model import SimpleNamespace, AutoGenerator

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    args   = _parse_args()
    device = args.gpu if (args.gpu != "cpu" and torch.cuda.is_available()) else "cpu"
    log.info(f"Device: {device}")

    output_path = args.output or os.path.join(args.model_dir, "fate_acc_results.csv")

    # ── Load data ──────────────────────────────────────────────────────────
    log.info(f"Loading adata: {args.data_path}")
    adata = sc.read_h5ad(args.data_path)

    log.info(f"Loading F_obs: {args.F_obs_path}")
    F_obs = pd.read_csv(args.F_obs_path, index_col=0)
    F_obs.index = F_obs.index.astype(str)
    log.info(f"  F_obs shape: {F_obs.shape}")

    tp_values  = sorted(adata.obs[args.tp_col].unique())
    tp_start   = float(args.tp_start) if args.tp_start is not None else tp_values[0]
    train_mask = adata.obs[args.well_col] != 2
    log.info(f"  Timepoints: {tp_values}  →  tp_start={tp_start}")
    log.info(f"  Train cells: {train_mask.sum()}  Test cells: {(~train_mask).sum()}")

    # ── Load model ─────────────────────────────────────────────────────────
    config_path = os.path.join(args.model_dir, "config.pt")
    train_pt    = os.path.join(args.model_dir, "train.best.pt")
    for fpath in (config_path, train_pt):
        if not os.path.exists(fpath):
            raise FileNotFoundError(fpath)

    config = SimpleNamespace(**torch.load(config_path, weights_only=False))
    net    = AutoGenerator(config)
    ckpt   = torch.load(train_pt, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])
    net    = net.to(device).eval()
    log.info(f"Model: x_dim={config.x_dim}  layers={config.layers}  k_dim={config.k_dim}")
    log.info(f"  num_steps = {int((config.train_t[-1]-config.start_t)/config.train_dt)}")

    # ── Evaluate ───────────────────────────────────────────────────────────
    result = evaluate_fate_accuracy(
        adata       = adata,
        net         = net,
        config      = config,
        F_obs       = F_obs,
        obsm_key    = args.obsm_key,
        n_dims      = args.n_dims,
        celltype_col= args.celltype_col,
        tp_col      = args.tp_col,
        tp_start    = tp_start,
        train_mask  = train_mask,
        n_sims      = args.n_sims,
        k           = args.k,
        device      = device,
    )

    log.info(f"\n{'='*50}")
    log.info(f"Fate Accuracy  : {result['accuracy']:.4f}")
    log.info(f"Pearson r      : {result['pearson_r']:.4f}  (p={result['pearson_p']:.3e})")
    log.info(f"Start cells    : {result['n_start_cells']}")
    log.info(f"{'='*50}")

    # ── Save summary CSV ───────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    df_summary = pd.DataFrame([{
        "accuracy":      result["accuracy"],
        "pearson_r":     result["pearson_r"],
        "pearson_p":     result["pearson_p"],
        "n_start_cells": result["n_start_cells"],
        "n_sims":        args.n_sims,
        "k_nn":          args.k,
        "obsm_key":      args.obsm_key,
        "n_dims":        args.n_dims,
        "model_dir":     args.model_dir,
    }])
    df_summary.to_csv(output_path, index=False)
    log.info(f"Summary → {output_path}")

    # Per-cell-type fate fractions
    F_hat = result["F_hat"]
    F_obs_al = result["F_obs_aligned"]
    fate_tbl = pd.DataFrame({
        "F_obs_mean": F_obs_al.mean(),
        "F_hat_mean": F_hat.mean(),
    })
    fate_csv = output_path.replace(".csv", "_celltype_fractions.csv")
    fate_tbl.to_csv(fate_csv)
    log.info(f"Cell-type fractions → {fate_csv}")
    log.info(f"\n{fate_tbl.to_string()}")

    # Per-cell F_hat
    fhat_csv = output_path.replace(".csv", "_F_hat.csv")
    F_hat.to_csv(fhat_csv)
    log.info(f"F_hat per cell → {fhat_csv}")


if __name__ == "__main__":
    main()
