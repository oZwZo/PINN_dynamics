"""
MIOFlow Training Script with GAGA (Geometric Autoencoder)
=========================================================
Pre-trains a GAGA autoencoder (via PHATE-based distance preservation),
then trains a MIOFlow Neural ODE in the GAGA latent space.

Uses the new ``mioflow`` package (MIOFlow class + fit_gaga).

Usage
-----
# PCA 50D input, GAGA → 2D latent, then MIOFlow ODE
python scripts/MIOFlow/01_train_with_gae.py \
    --data_path data/klein_addpop.h5ad \
    --obsm_key X_pca \
    --obs_time_key Time_point \
    --exp_name klein_pca_gaga

# DM 10D input, GAGA → 2D latent
python scripts/MIOFlow/01_train_with_gae.py \
    --data_path data/klein_addpop.h5ad \
    --obsm_key DM_EigenVectors --n_dims 10 \
    --obs_time_key Time_point \
    --exp_name klein_dm10_gaga

# With PHATE pre-computed in adata.obsm['X_phate']
python scripts/MIOFlow/01_train_with_gae.py \
    --data_path data/klein_addpop.h5ad \
    --obsm_key X_pca --phate_key X_phate \
    --obs_time_key Time_point
"""

import os
import sys
import json
import time
import argparse
import logging

# ── tqdm monkey-patch (MIOFlow imports from tqdm.notebook) ─────────────────
import tqdm as _tqdm
import tqdm.notebook
tqdm.notebook.tqdm = _tqdm.tqdm

import numpy as np
import torch
import scanpy as sc

from mioflow.gaga import Autoencoder, fit_gaga, train_gaga_two_phase, dataloader_from_pc
from mioflow.mioflow import MIOFlow
from mioflow.core.models.ode_model import ODEFunc
from mioflow.core.models.sde_model import SDEFunc

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train MIOFlow with GAGA autoencoder")

    # ── Data ──────────────────────────────────────────────────────────────
    p.add_argument("--data_path", required=True, help="Path to .h5ad file")
    p.add_argument("--obsm_key", default="X_pca",
                   help="obsm key for GAGA input (default: X_pca)")
    p.add_argument("--n_dims", type=int, default=None,
                   help="Number of dimensions to retain from obsm_key (default: all)")
    p.add_argument("--obs_time_key", default="Time_point",
                   help="Column in adata.obs for timepoints (default: Time_point)")
    p.add_argument("--well_col", default="Well",
                   help="Column for train/test split (default: Well)")
    p.add_argument("--phate_key", default=None,
                   help="obsm key for pre-computed PHATE; if unset, PHATE is computed from obsm_key")

    # ── Output ────────────────────────────────────────────────────────────
    p.add_argument("--exp_name", default=None,
                   help="Experiment name (default: auto-generated)")
    p.add_argument("--output_dir", default="logs/MIOFlow",
                   help="Root output directory (default: results/MIOFlow)")

    # ── GAGA hyperparameters ──────────────────────────────────────────────
    p.add_argument("--gaga_latent_dim", type=int, default=2,
                   help="GAGA latent dimension (default: 2)")
    p.add_argument("--gaga_hidden_dims", default="128,64",
                   help="GAGA hidden layers, comma-separated (default: 128,64)")
    p.add_argument("--gaga_batch_size", type=int, default=1024,
                   help="GAGA dataloader batch size (default: 1024)")
    p.add_argument("--gaga_encoder_epochs", type=int, default=10,
                   help="GAGA phase-1 epochs: distance preservation (default: 10)")
    p.add_argument("--gaga_decoder_epochs", type=int, default=10,
                   help="GAGA phase-2 epochs: reconstruction (default: 10)")
    p.add_argument("--gaga_lr", type=float, default=1e-3,
                   help="GAGA learning rate (default: 1e-3)")
    p.add_argument("--phate_n_components", type=int, default=2,
                   help="Number of PHATE components if computing from scratch (default: 2)")
    p.add_argument("--phate_n_pcs", type=int, default=30,
                   help="Number of PCs to use as PHATE input (default: 30)")

    # ── MIOFlow ODE hyperparameters ───────────────────────────────────────
    p.add_argument("--hidden_dim", type=int, default=128,
                   help="ODE hidden dimension (default: 128)")
    p.add_argument("--n_epochs", type=int, default=100,
                   help="MIOFlow training epochs (default: 100)")
    p.add_argument("--lambda_ot", type=float, default=1.0)
    p.add_argument("--use_density_loss", action="store_true", default=True)
    p.add_argument("--no_density_loss", dest="use_density_loss", action="store_false")
    p.add_argument("--lambda_density", type=float, default=0.1)
    p.add_argument("--lambda_energy", type=float, default=0.01)
    p.add_argument("--energy_time_steps", type=int, default=20)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--sample_size", type=int, default=100)
    p.add_argument("--n_trajectories", type=int, default=100)
    p.add_argument("--n_bins", type=int, default=100)
    p.add_argument("--momentum_beta", type=float, default=0.0)

    # ── SDE options ───────────────────────────────────────────────────────
    p.add_argument("--use_sde", action="store_true", default=False)
    p.add_argument("--diffusion_scale", type=float, default=0.1)
    p.add_argument("--sde_dt", type=float, default=0.1)

    # ── Misc ──────────────────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # ── Experiment directory ──────────────────────────────────────────────
    exp_name = args.exp_name or f"mioflow_{args.obsm_key}_gaga"
    exp_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    log.info(f"Experiment: {exp_name}  →  {exp_dir}")

    # ── Seed ──────────────────────────────────────────────────────────────
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Load data ─────────────────────────────────────────────────────────
    log.info(f"Loading {args.data_path}")
    adata = sc.read_h5ad(args.data_path)
    log.info(f"  {adata.n_obs} cells, {adata.n_vars} genes")

    # Train subset (for GAGA + PHATE)
    train_mask = adata.obs[args.well_col].astype(int) != 2
    train_ad = adata[train_mask].copy()
    log.info(f"  Train cells: {train_ad.n_obs}")

    # ── Prepare obsm input ────────────────────────────────────────────────
    X_raw = train_ad.obsm[args.obsm_key]
    if args.n_dims is not None:
        X_raw = X_raw[:, :args.n_dims]
    X_raw = np.asarray(X_raw, dtype=np.float32)
    input_dim = X_raw.shape[1]
    log.info(f"  obsm_key={args.obsm_key}  dims={input_dim}")

    # ── PHATE ─────────────────────────────────────────────────────────────
    if args.phate_key and args.phate_key in train_ad.obsm:
        log.info(f"  Using pre-computed PHATE from obsm['{args.phate_key}']")
        X_phate = np.asarray(train_ad.obsm[args.phate_key], dtype=np.float32)
    else:
        import phate
        log.info(f"  Computing PHATE ({args.phate_n_components}D from {args.phate_n_pcs} PCs)")
        phate_input = train_ad.obsm[args.obsm_key][:, :args.phate_n_pcs]
        phate_op = phate.PHATE(n_components=args.phate_n_components, n_jobs=-2)
        X_phate = phate_op.fit_transform(phate_input).astype(np.float32)

    # ── Train GAGA ────────────────────────────────────────────────────────
    gaga_hidden_dims = [int(x) for x in args.gaga_hidden_dims.split(",")]
    latent_dim = args.gaga_latent_dim

    log.info(f"  GAGA: {input_dim} → {gaga_hidden_dims} → {latent_dim}")
    gaga_start = time.time()
    gaga_model = fit_gaga(
        X_pca=X_raw,
        X_phate=X_phate,
        latent_dim=latent_dim,
        hidden_dims=gaga_hidden_dims,
        batch_size=args.gaga_batch_size,
        encoder_epochs=args.gaga_encoder_epochs,
        decoder_epochs=args.gaga_decoder_epochs,
        learning_rate=args.gaga_lr,
    )
    gaga_time = time.time() - gaga_start
    log.info(f"  GAGA training done in {gaga_time:.1f}s")

    # ── Save GAGA checkpoint ──────────────────────────────────────────────
    gaga_ckpt = {
        "model_state_dict": gaga_model.state_dict(),
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "hidden_dims": gaga_hidden_dims,
        "scaler_pca": gaga_model.input_scaler,
    }
    gaga_ckpt_path = os.path.join(exp_dir, "gaga_checkpoint.pt")
    torch.save(gaga_ckpt, gaga_ckpt_path)
    log.info(f"  GAGA checkpoint → {gaga_ckpt_path}")

    # ── Train MIOFlow ─────────────────────────────────────────────────────
    log.info(f"Initialising MIOFlow: hidden_dim={args.hidden_dim}, n_epochs={args.n_epochs}")
    mf = MIOFlow(
        adata,
        gaga_model=gaga_model.to("cpu"),
        gaga_input_key=args.obsm_key,
        obs_time_key=args.obs_time_key,
        debug_level="info",
        hidden_dim=args.hidden_dim,
        use_cuda=torch.cuda.is_available(),
        momentum_beta=args.momentum_beta,
        n_epochs=args.n_epochs,
        use_density_loss=args.use_density_loss,
        lambda_ot=args.lambda_ot,
        lambda_density=args.lambda_density,
        lambda_energy=args.lambda_energy,
        energy_time_steps=args.energy_time_steps,
        learning_rate=args.learning_rate,
        sample_size=args.sample_size,
        n_trajectories=args.n_trajectories,
        n_bins=args.n_bins,
        exp_dir=exp_dir,
        # SDE
        use_sde=args.use_sde,
        diffusion_scale=args.diffusion_scale,
        sde_dt=args.sde_dt,
    )

    ode_start = time.time()
    mf.fit()
    ode_time = time.time() - ode_start
    log.info(f"MIOFlow training done in {ode_time:.1f}s  (total: {gaga_time + ode_time:.1f}s)")

    # ── Save full checkpoint ──────────────────────────────────────────────
    ckpt = {
        # ODE
        "ode_state_dict": mf.ode_model.state_dict(),
        "ode_class": type(mf.ode_model).__name__,
        "hidden_dim": mf.hidden_dim,
        "ode_input_dim": mf.ode_model.model[0].in_features,
        # Normalization
        "mean_vals": mf.mean_vals,
        "std_vals": mf.std_vals,
        # GAGA
        "gaga_state_dict": gaga_model.state_dict(),
        "gaga_input_dim": input_dim,
        "gaga_latent_dim": latent_dim,
        "gaga_hidden_dims": gaga_hidden_dims,
        "scaler_pca": gaga_model.input_scaler,
        # Losses
        "losses": mf.losses,
    }
    ckpt_path = os.path.join(exp_dir, "mioflow_checkpoint.pt")
    torch.save(ckpt, ckpt_path)
    log.info(f"Checkpoint → {ckpt_path}")

    # ── Save options ──────────────────────────────────────────────────────
    opts = {k: v for k, v in vars(args).items()}
    opts["gaga_hidden_dims"] = gaga_hidden_dims
    opts["gaga_time_s"] = round(gaga_time, 1)
    opts["ode_time_s"] = round(ode_time, 1)
    opts["exp_dir"] = exp_dir
    with open(os.path.join(exp_dir, "options.json"), "w") as f:
        json.dump(opts, f, indent=2)

    log.info(f"Done. Results at: {exp_dir}")


def load_checkpoint(ckpt_path, adata, obsm_key="X_pca", obs_time_key="Time_point"):
    """Reload a trained MIOFlow + GAGA from a checkpoint file.

    Returns (mf, gaga_model) with mf.is_fitted = True, ready for prediction.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Rebuild GAGA
    gaga_model = Autoencoder(
        ckpt["gaga_input_dim"], ckpt["gaga_latent_dim"],
        hidden_dims=ckpt["gaga_hidden_dims"],
    )
    gaga_model.load_state_dict(ckpt["gaga_state_dict"])
    gaga_model.input_scaler = ckpt["scaler_pca"]

    # Rebuild MIOFlow (triggers _prepare_data internally)
    mf = MIOFlow(
        adata,
        gaga_model=gaga_model,
        gaga_input_key=obsm_key,
        obs_time_key=obs_time_key,
        hidden_dim=ckpt["hidden_dim"],
    )

    # Restore ODE weights + normalization
    mf.mean_vals = ckpt["mean_vals"]
    mf.std_vals = ckpt["std_vals"]

    OdeClass = ODEFunc if ckpt["ode_class"] == "ODEFunc" else SDEFunc
    mf.ode_model = OdeClass(
        input_dim=ckpt["ode_input_dim"], hidden_dim=ckpt["hidden_dim"],
    )
    mf.ode_model.load_state_dict(ckpt["ode_state_dict"])
    mf.losses = ckpt.get("losses")
    mf.is_fitted = True
    return mf, gaga_model


if __name__ == "__main__":
    main()
