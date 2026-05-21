"""
MIOFlow Combined Evaluation Script (with GAGA)
===============================================
Evaluates a GAGA-trained MIOFlow model on held-out test cells (Well == 2).
Computes all metrics in one run: per-cell fate accuracy, Pearson r,
W2 (population + per-clone).

Uses the new ``mioflow`` package: GAGA encodes start cells, ODE simulates
in latent space, GAGA decodes endpoints back to the original obsm space.

Fate:  F_obs cells (tp_first) → simulate → KNN classify → F_hat
W2:    Clone-filtered test cells (tp_mid) → simulate → compare to tp_last

Outputs:
  - eval_combined.csv          (one row, sim_mode=ode)
  - F_hat_ode.csv              (per-cell fate predictions)
  - w2_per_clone_ode.csv       (per-clone W2)

Usage
-----
python scripts/MIOFlow/03_evaluate_with_gae.py \
    --data_path data/klein_addpop.h5ad \
    --model_dir logs/MIOFlow/klein_pca_gaga_run1 \
    --obsm_key X_pca \
    --obs_time_key Time_point \
    --device cuda

# With explicit checkpoint file
python scripts/MIOFlow/03_evaluate_with_gae.py \
    --data_path data/klein_addpop.h5ad \
    --model_dir logs/MIOFlow/klein_pca_gaga \
    --ckpt_name mioflow_checkpoint.pt \
    --obsm_key X_pca \
    --device cpu
"""

import os
import sys
import argparse
import logging
# ── tqdm monkey-patch (required before mioflow imports) ────────────────────
import tqdm as _tqdm
import tqdm.notebook
tqdm.notebook.tqdm = _tqdm.tqdm

import numpy as np
import pandas as pd
import torch
import scanpy as sc
from sklearn.neighbors import KNeighborsClassifier
from torchdiffeq import odeint

from mioflow.gaga import Autoencoder
from mioflow.mioflow import MIOFlow
from mioflow.core.models.ode_model import ODEFunc
from mioflow.core.models.sde_model import SDEFunc

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import fate_eval_pipeline as fate_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ─── Simulation via MIOFlow object ──────────────────────────────────────────

def simulate_mf(mf, start_barcodes, start_time, end_time, adata,
                obsm_key="X_pca", n_steps=100, device="cuda:0"):
    """Simulate start cells forward through the MIOFlow ODE.

    Encodes barcoded cells via GAGA, runs ODE in normalised latent space,
    decodes back to the original obsm space (e.g. X_pca).

    Returns ndarray (n_cells, n_dims) — endpoints in obsm_key space.
    """
    fate_ad = adata[start_barcodes].copy()
    X_raw = np.asarray(fate_ad.obsm[obsm_key], dtype=np.float32)

    scaler = mf.gaga_autoencoder.input_scaler
    X_scaled = scaler.transform(X_raw) if scaler is not None else X_raw

    mf.gaga_autoencoder.eval()
    with torch.no_grad():
        embedding = mf.gaga_autoencoder.encode(
            torch.tensor(X_scaled)
        ).cpu().numpy().astype(np.float64)

    normed = (embedding - mf.mean_vals) / mf.std_vals

    t_bins = torch.linspace(start_time, end_time, n_steps, device=device)
    mf.ode_model.eval()
    with torch.no_grad():
        X_ts = torch.tensor(normed, dtype=torch.float32, device=device)
        traj = odeint(mf.ode_model.to(device), X_ts, t_bins)
        # Inverse the (emb - mean)/std applied at line 88 before decoding;
        # the GAGA decoder was trained on un-normalized latent (mioflow.py:328
        # canonical generate path applies *std + mean before decode).
        std_t = torch.tensor(mf.std_vals, dtype=torch.float32, device=device)
        mean_t = torch.tensor(mf.mean_vals, dtype=torch.float32, device=device)
        traj_last_unnormed = traj[-1] * std_t + mean_t
        decoded = mf.gaga_autoencoder.to(device).decode(traj_last_unnormed)

    decoded_np = decoded.cpu().numpy()
    mf.gaga_autoencoder.cpu()
    mf.ode_model.cpu()

    endpoint = scaler.inverse_transform(decoded_np) if scaler is not None else decoded_np
    return endpoint


def _make_sim_fn(mf, adata, obsm_key, start_time, end_time, n_steps=100):
    """Build a fate_eval_pipeline-compatible sim_fn from a MIOFlow object."""

    def _simulate(start_cells, model, n_sims, device):
        # model is (mf, start_barcodes) tuple
        mf_obj, barcodes = model
        return simulate_mf(
            mf_obj, barcodes, start_time, end_time, adata,
            obsm_key=obsm_key, n_steps=n_steps, device=device,
        )

    return _simulate


# ─── Checkpoint loading ─────────────────────────────────────────────────────

def load_checkpoint(ckpt_path, adata, obsm_key="X_pca", obs_time_key="Time_point"):
    """Reload a trained MIOFlow + GAGA from a checkpoint file."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    gaga_model = Autoencoder(
        ckpt["gaga_input_dim"], ckpt["gaga_latent_dim"],
        hidden_dims=ckpt["gaga_hidden_dims"],
    )
    gaga_model.load_state_dict(ckpt["gaga_state_dict"])
    gaga_model.input_scaler = ckpt["scaler_pca"]

    mf = MIOFlow(
        adata, gaga_model=gaga_model,
        gaga_input_key=obsm_key, obs_time_key=obs_time_key,
        hidden_dim=ckpt["hidden_dim"],
    )
    mf.mean_vals = ckpt["mean_vals"]
    mf.std_vals = ckpt["std_vals"]

    OdeClass = ODEFunc if ckpt["ode_class"] == "ODEFunc" else SDEFunc

    input_dim = ckpt["ode_input_dim"]-1 if "ode_input_dim" in ckpt else ckpt["input_dim"]-1
    mf.ode_model = OdeClass(
        input_dim=input_dim, hidden_dim=ckpt["hidden_dim"],
    )
    mf.ode_model.load_state_dict(ckpt["ode_state_dict"])
    mf.losses = ckpt.get("losses")
    mf.is_fitted = True
    return mf, gaga_model


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Combined evaluation for GAGA-trained MIOFlow (fate accuracy + W2)"
    )
    p.add_argument("--data_path", required=True, help="Path to .h5ad file")
    p.add_argument("--model_dir", required=True,
                   help="Dir containing mioflow_checkpoint.pt")
    p.add_argument("--ckpt_name", default="mioflow_checkpoint.pt",
                   help="Checkpoint filename (default: mioflow_checkpoint.pt)")
    p.add_argument("--F_obs_path", default="data/klein/F_obs.csv",
                   help="Ground-truth fate proportions CSV")
    p.add_argument("--clone_proportions", default="data/klein/clone_proportions.csv",
                   help="Clone proportions CSV")
    p.add_argument("--obsm_key", default="X_pca",
                   help="obsm key used during training")
    p.add_argument("--n_dims", type=int, default=None,
                   help="Embedding dimensions (default: all)")
    p.add_argument("--obs_time_key", default="Time_point",
                   help="Timepoint column in adata.obs")
    p.add_argument("--tp_col", default="timepoint_tx_days",
                   help="Timepoint column for W2 src/tgt filtering")
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

    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"
    log.info(f"Device: {device}")

    output_dir = args.output_dir or args.model_dir
    os.makedirs(output_dir, exist_ok=True)

    obsm_key = args.obsm_key

    # ── Load adata ────────────────────────────────────────────────────────
    log.info(f"Loading adata: {args.data_path}")
    adata = sc.read_h5ad(args.data_path)

    tp_values = sorted(adata.obs[args.tp_col].unique())
    tp_first = tp_values[0]
    tp_mid = tp_values[1]
    tp_last = tp_values[-1]
    log.info(f"  Timepoints: {tp_values} (first={tp_first}, mid={tp_mid}, last={tp_last})")

    # ── Load checkpoint ───────────────────────────────────────────────────
    ckpt_path = os.path.join(args.model_dir, args.ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    mf, gaga_model = load_checkpoint(
        ckpt_path, adata,
        obsm_key=obsm_key,
        obs_time_key=args.obs_time_key,
    )
    log.info(f"Model loaded: ode_input_dim={mf.ode_model.model[0].in_features}, "
             f"hidden_dim={mf.hidden_dim}")

    # ── Train / test split ────────────────────────────────────────────────
    train_mask = adata.obs[args.well_col].astype(int) != 2
    adata_train = adata[train_mask]
    adata_test = adata[~train_mask]

    # ── Load ground truth ─────────────────────────────────────────────────
    F_obs = pd.read_csv(args.F_obs_path, index_col=0)
    F_obs.index = F_obs.index.astype(str)
    F_obs = F_obs[F_obs.index.isin(adata.obs_names.astype(str))]
    clone_proportions = pd.read_csv(args.clone_proportions, index_col=0)

    # ═══════════════════════════════════════════════════════════════════════
    # FATE EVALUATION
    # ═══════════════════════════════════════════════════════════════════════
    fobs_idx = F_obs.index.astype(str)
    valid_mask = adata.obs_names.astype(str).isin(fobs_idx)
    start_barcodes_fate = list(adata[valid_mask].obs_names.astype(str))
    log.info(f"Fate start cells: {len(start_barcodes_fate)} (matched from F_obs)")

    # KNN reference: all train cells
    x_ref = np.asarray(adata_train.obsm[obsm_key], dtype=np.float32)
    if args.n_dims is not None:
        x_ref = x_ref[:, :args.n_dims]
    y_ref = adata_train.obs[args.celltype_col].values.astype(str)
    log.info(f"KNN reference (all train cells): {len(x_ref)} cells")

    # Simulate fate
    fate_endpoint = simulate_mf(
        mf, start_barcodes_fate,
        start_time=0.0, end_time=4.0,#float(len(tp_values) - 1),
        adata=adata, obsm_key=obsm_key, device=device,
    )
    if args.n_dims is not None:
        fate_endpoint = fate_endpoint[:, :args.n_dims]

    # KNN classify
    knn = KNeighborsClassifier(n_neighbors=args.k, metric="euclidean")
    knn.fit(x_ref, y_ref)

    # Build F_hat
    cell_types = sorted(F_obs.columns)
    F_hat = pd.DataFrame(0.0, index=start_barcodes_fate, columns=cell_types)
    y_pred = knn.predict(fate_endpoint)
    for i, bc in enumerate(start_barcodes_fate):
        pred_type = y_pred[i]
        if pred_type in cell_types:
            F_hat.loc[bc, pred_type] = 1.0

    # Align and compute accuracy
    shared = sorted(set(F_hat.index) & set(F_obs.index))
    F_obs_aligned = F_obs.loc[shared, cell_types]
    F_hat_aligned = F_hat.loc[shared, cell_types]

    y_true = F_obs_aligned.idxmax(axis=1).values
    y_pred_aligned = F_hat_aligned.idxmax(axis=1).values

    import sklearn.metrics
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred_aligned)

    # Pearson r (population-level per cell-type)
    from scipy.stats import pearsonr
    f_obs_mean = F_obs_aligned.mean(axis=0)
    f_hat_mean = F_hat_aligned.mean(axis=0)
    pearson_r, _ = pearsonr(f_obs_mean.values, f_hat_mean.values)

    log.info(f"  Fate accuracy: {accuracy:.4f}")
    log.info(f"  Fate Pearson r: {pearson_r:.4f}")
    F_hat.to_csv(os.path.join(output_dir, "F_hat_ode.csv"))

    # ═══════════════════════════════════════════════════════════════════════
    # W2 EVALUATION
    # ═══════════════════════════════════════════════════════════════════════
    w2_clone_mask = adata_test.obs.clones.isin(clone_proportions.index)
    w2_test_ad = adata_test[w2_clone_mask]

    src_ad_w2 = w2_test_ad[w2_test_ad.obs[args.tp_col] == tp_mid]
    tgt_ad_w2 = w2_test_ad[w2_test_ad.obs[args.tp_col] == tp_last]
    log.info(f"W2 src (tp{tp_mid}): {src_ad_w2.n_obs}, tgt (tp{tp_last}): {tgt_ad_w2.n_obs}")

    # Simulate W2 source cells
    # Timepoint indices: first=0, mid=1, last=2 (for 3-timepoint data)
    tp_idx = {v: i for i, v in enumerate(tp_values)}
    w2_endpoint = simulate_mf(
        mf, list(src_ad_w2.obs_names.astype(str)),
        start_time=float(tp_idx[tp_mid]),
        end_time=float(tp_idx[tp_last]),
        adata=adata, obsm_key=obsm_key, device=device,
    )
    w2_true = np.asarray(tgt_ad_w2.obsm[obsm_key], dtype=np.float32)
    if args.n_dims is not None:
        w2_endpoint = w2_endpoint[:, :args.n_dims]
        w2_true = w2_true[:, :args.n_dims]

    # Population W2
    w2_raw = fate_eval.compute_w2(w2_endpoint, w2_true, device=device)
    log.info(f"  W2 raw: {w2_raw:.4f}")

    # Per-clone W2
    src_clones = src_ad_w2.obs.clones.values
    tgt_clones = tgt_ad_w2.obs.clones.values
    shared_clones = sorted(set(src_clones) & set(tgt_clones))

    clone_records = []
    for clone in shared_clones:
        src_mask = src_clones == clone
        tgt_mask = tgt_clones == clone
        if src_mask.sum() < 2 or tgt_mask.sum() < 2:
            continue
        clone_ep = w2_endpoint[src_mask]
        clone_tgt = w2_true[tgt_mask]
        cw2 = fate_eval.compute_w2(clone_ep, clone_tgt, device=device)
        clone_records.append({
            "clone": clone,
            "w2_raw": float(cw2),
            "clone_size_src": int(src_mask.sum()),
            "clone_size_tgt": int(tgt_mask.sum()),
        })

    df_clone_w2 = pd.DataFrame(clone_records)
    df_clone_w2.to_csv(os.path.join(output_dir, "w2_per_clone_ode.csv"), index=False)
    log.info(f"  Per-clone W2: {len(df_clone_w2)} clones")

    # ═══════════════════════════════════════════════════════════════════════
    # SAVE eval_combined.csv
    # ═══════════════════════════════════════════════════════════════════════
    df_combined = pd.DataFrame([{
        "sim_mode": "ode",
        "accuracy": accuracy,
        "pearson_r": pearson_r,
        "w2_raw": float(w2_raw),
        "n_start_cells": len(start_barcodes_fate),
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
