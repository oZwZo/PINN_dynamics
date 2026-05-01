"""
MIOFlow Training Script
========================
Trains a MIOFlow Neural ODE model on klein_addpop.h5ad.

Usage
-----
# Experiment 1: PCA 30D
python scripts/MIOFlow/01_train.py \
    --data_path data/klein/klein_addpop.h5ad \
    --obsm_key X_pca --n_dims 30 \
    --config pca --exp_name klein_addpop_pca30 --seed 10

# Experiment 2: DM 10D
python scripts/MIOFlow/01_train.py \
    --data_path data/klein/klein_addpop.h5ad \
    --obsm_key DM_EigenVectors --n_dims 10 \
    --config dm --exp_name klein_addpop_dm10 --seed 10
"""

import os
import sys
import json
import time
import argparse
import logging

# ── tqdm monkey-patch (MIOFlow train.py imports from tqdm.notebook) ───────
import tqdm as _tqdm
import tqdm.notebook
tqdm.notebook.tqdm = _tqdm.tqdm

import numpy as np
import pandas as pd
import torch
import scanpy as sc

from MIOFlow.utils import set_seeds, config_criterion
from MIOFlow.models import make_model
from MIOFlow.train import training_regimen
from MIOFlow.plots import plot_losses
from MIOFlow.eval import generate_plot_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── per-config hyperparameter presets ─────────────────────────────────────
CONFIG_DEFAULTS = {
    "pca": dict(
        layers=[128, 128],
        activation="CELU",
        n_local_epochs=40,
        n_epochs=300,
        n_post_local_epochs=0,
        sample_size=100,
        lambda_density=20,
        criterion_name="ot",
    ),
    "dm": dict(
        layers=[128,128],
        activation="CELU",
        n_local_epochs=40,
        n_epochs=300,
        n_post_local_epochs=0,
        sample_size=100,
        lambda_density=20,
        criterion_name="ot",
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description="Train a MIOFlow model")
    p.add_argument("--data_path", required=True, help="Path to .h5ad file")
    p.add_argument("--obsm_key", default="X_pca", help="obsm key for embedding (default: X_pca)")
    p.add_argument("--n_dims", type=int, default=30, help="Number of dimensions to retain (default: 30)")
    p.add_argument("--config", choices=["pca", "dm"], default=None,
                   help="Preset config: 'pca' or 'dm'")
    p.add_argument("--exp_name", default=None,
                   help="Experiment name for output directory (default: auto-generated)")
    p.add_argument("--output_dir", default="results/MIOFlow",
                   help="Root output directory (default: results/MIOFlow)")
    p.add_argument("--tp_col", default="timepoint_tx_days",
                   help="Timepoint column in adata.obs (default: timepoint_tx_days)")
    p.add_argument("--well_col", default="Well",
                   help="Well column for train/test split (default: Well)")
    p.add_argument("--seed", type=int, default=10, help="Random seed (default: 10)")
    # Architecture overrides
    p.add_argument("--layers", default=None,
                   help="Hidden layers as comma-separated ints, e.g. '64,128,64'")
    p.add_argument("--activation", default=None, help="Activation function (default: from config)")
    # Training overrides
    p.add_argument("--n_local_epochs", type=int, default=None)
    p.add_argument("--n_epochs", type=int, default=None)
    p.add_argument("--n_post_local_epochs", type=int, default=None)
    p.add_argument("--lambda_density", type=float, default=None)
    p.add_argument("--sample_size", type=int, default=None)
    p.add_argument("--criterion_name", default=None, choices=["ot", "mmd"])
    p.add_argument("--plot_every", type=int, default=20,
                   help="Plot comparison every N epochs (default: 20, 0 to disable)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── resolve hyperparameters ───────────────────────────────────────────
    hp = {}
    if args.config is not None:
        hp.update(CONFIG_DEFAULTS[args.config])

    # CLI overrides
    if args.layers is not None:
        hp["layers"] = [int(x) for x in args.layers.split(",")]
    if args.activation is not None:
        hp["activation"] = args.activation
    if args.n_local_epochs is not None:
        hp["n_local_epochs"] = args.n_local_epochs
    if args.n_epochs is not None:
        hp["n_epochs"] = args.n_epochs
    if args.n_post_local_epochs is not None:
        hp["n_post_local_epochs"] = args.n_post_local_epochs
    if args.lambda_density is not None:
        hp["lambda_density"] = args.lambda_density
    if args.sample_size is not None:
        hp["sample_size"] = args.sample_size
    if args.criterion_name is not None:
        hp["criterion_name"] = args.criterion_name

    for key in ("layers", "activation", "n_local_epochs", "n_epochs", "criterion_name"):
        if key not in hp:
            raise ValueError(f"--{key} must be set (or use --config pca/dm to apply a preset)")

    hp.setdefault("n_post_local_epochs", 0)
    hp.setdefault("lambda_density", 20)
    hp.setdefault("sample_size", 100)

    # ── experiment name & directory ───────────────────────────────────────
    exp_name = args.exp_name
    if exp_name is None:
        exp_name = f"klein_{args.obsm_key}_{args.n_dims}d"
    exp_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    log.info(f"Experiment: {exp_name}")
    log.info(f"Output dir: {exp_dir}")

    # ── load data ─────────────────────────────────────────────────────────
    log.info(f"Loading adata from {args.data_path}")
    adata = sc.read_h5ad(args.data_path)

    train_mask = adata.obs[args.well_col].astype(int) != 2
    adata_train = adata[train_mask]
    log.info(f"  Total cells: {adata.n_obs}  Train cells: {adata_train.n_obs}")

    # ── build MIOFlow DataFrame ───────────────────────────────────────────
    embedding = np.array(adata_train.obsm[args.obsm_key][:, :args.n_dims])
    timepoints = np.array(adata_train.obs[args.tp_col].values)

    n_dims = args.n_dims
    df = pd.DataFrame(embedding, columns=[f"d{i}" for i in range(1, n_dims + 1)])
    df["samples"] = timepoints.astype(np.int32)

    model_features = n_dims
    groups = sorted(df["samples"].unique())
    log.info(f"  Embedding: {args.obsm_key}  Dims: {n_dims}  Groups: {groups}")
    log.info(f"  DataFrame shape: {df.shape}")

    # ── set seeds and configure model ─────────────────────────────────────
    set_seeds(args.seed)
    use_cuda = torch.cuda.is_available()
    log.info(f"  CUDA available: {use_cuda}")

    model = make_model(
        model_features, hp["layers"],
        activation=hp["activation"], scales=None, use_cuda=use_cuda
    )
    log.info(f"  Model: ToyODE  layers={hp['layers']}  activation={hp['activation']}")

    criterion = config_criterion(hp["criterion_name"])
    optimizer = torch.optim.AdamW(model.parameters())

    # ── save options ──────────────────────────────────────────────────────
    opts = {
        "data_path": os.path.abspath(args.data_path),
        "obsm_key": args.obsm_key,
        "n_dims": n_dims,
        "model_features": model_features,
        "groups": [int(g) for g in groups],
        "layers": hp["layers"],
        "activation": hp["activation"],
        "n_local_epochs": hp["n_local_epochs"],
        "n_epochs": hp["n_epochs"],
        "n_post_local_epochs": hp["n_post_local_epochs"],
        "criterion_name": hp["criterion_name"],
        "lambda_density": hp["lambda_density"],
        "sample_size": hp["sample_size"],
        "seed": args.seed,
        "use_cuda": use_cuda,
        "use_emb": False,
        "use_gae": False,
        "hold_one_out": False,
        "hold_out": 3,
        "reverse_schema": False,
        "reverse_n": 2,
        "sde_scales": None,
        "tp_col": args.tp_col,
        "well_col": args.well_col,
        "exp_name": exp_name,
    }
    with open(os.path.join(exp_dir, "options.json"), "w") as f:
        json.dump(opts, f, indent=2)

    # ── train ─────────────────────────────────────────────────────────────
    plot_every = args.plot_every if args.plot_every > 0 else None
    log.info(f"\nStarting training: {hp['n_local_epochs']} local + {hp['n_epochs']} global + {hp['n_post_local_epochs']} post-local epochs")

    # Workaround: MIOFlow's train() indexes ot_lambda_global with loop
    # indices (0,1,2,...) but defaults to group-value keys ({2:1, 4:1, 6:1}).
    # Pass index-keyed dict explicitly to avoid KeyError.
    _ot_lambda = {i: 1.0 for i in range(len(groups))}

    start_time = time.time()
    local_losses, batch_losses, globe_losses = training_regimen(
        n_local_epochs=hp["n_local_epochs"],
        n_epochs=hp["n_epochs"],
        n_post_local_epochs=hp["n_post_local_epochs"],
        exp_dir=exp_dir,
        model=model, df=df, groups=groups, optimizer=optimizer,
        criterion=criterion, use_cuda=use_cuda,
        hold_one_out=False, hold_out=3,
        use_density_loss=True, lambda_density=hp["lambda_density"],
        autoencoder=None, use_emb=False, use_gae=False,
        sample_size=(hp["sample_size"],),
        logger=None,
        reverse_schema=False, reverse_n=2,
        plot_every=plot_every,
        n_points=1000, n_trajectories=100, n_bins=100,
        ot_lambda_global=_ot_lambda,
    )
    run_time = time.time() - start_time
    log.info(f"Training complete in {run_time:.1f}s")

    # ── save checkpoint ───────────────────────────────────────────────────
    ckpt = {
        "model_state_dict": model.state_dict(),
        "autoencoder_state_dict": None,
        "options": opts,
    }
    ckpt_path = os.path.join(exp_dir, "model_checkpoints.pt")
    torch.save(ckpt, ckpt_path)
    log.info(f"Checkpoint saved -> {ckpt_path}")

    # ── save losses ───────────────────────────────────────────────────────
    np.save(os.path.join(exp_dir, "batch_losses.npy"), np.array(batch_losses))
    np.save(os.path.join(exp_dir, "globe_losses.npy"), np.array(globe_losses))

    try:
        plot_losses(
            local_losses, batch_losses, globe_losses,
            save=True, path=exp_dir, file="losses.png"
        )
        log.info(f"Loss plot saved -> {exp_dir}/losses.png")
    except Exception as e:
        log.warning(f"Could not save loss plot: {e}")

    # ── generate final comparison ─────────────────────────────────────────
    try:
        generated, trajectories = generate_plot_data(
            model, df, 1000, 100, 100,
            use_cuda=use_cuda, samples_key="samples",
            autoencoder=None, recon=False
        )
        np.save(os.path.join(exp_dir, "generated.npy"), generated)
        np.save(os.path.join(exp_dir, "trajectories.npy"), trajectories)
        log.info(f"Generated points shape: {generated.shape}")
    except Exception as e:
        log.warning(f"Could not generate final points: {e}")

    log.info(f"\nDone. Results at: {exp_dir}")


if __name__ == "__main__":
    main()
