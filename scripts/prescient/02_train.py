"""
PRESCIENT Training Script
==========================
Wraps PRESCIENT train_model with sensible defaults for the PCA and DM configs.
Calls PRESCIENT as a Python import (not subprocess) so that torch.load/save
monkey-patches for PyTorch >= 2.6 compatibility take effect.

Usage
-----
# PCA config (30-dim, k_dim=500, layers=1)
python scripts/prescient/02_train.py \
    --data_path results/PRESCIENT/pca_run/data.pt \
    --out_dir   results/PRESCIENT/pca_run/ \
    --config    pca \
    --seed      2 \
    --gpu       0

# DM config (10-dim, k_dim=64, layers=4)
python scripts/prescient/02_train.py \
    --data_path results/PRESCIENT/dm_run/data.pt \
    --out_dir   results/PRESCIENT/dm_run/ \
    --config    dm \
    --seed      2 \
    --gpu       0
"""

import os
import argparse
import logging

import torch

# ── PyTorch >= 2.6 compat ────────────────────────────────────────────────
# PRESCIENT internally uses torch.load() / torch.save() without the flags
# needed for PyTorch >= 2.6.  Monkey-patch BEFORE importing prescient.
_original_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(f, *args, **kwargs)
torch.load = _patched_torch_load

_original_torch_save = torch.save
def _patched_torch_save(obj, f, *args, **kwargs):
    kwargs.setdefault("pickle_protocol", 4)
    return _original_torch_save(obj, f, *args, **kwargs)
torch.save = _patched_torch_save

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── per-config hyperparameter presets ──────────────────────────────────────
CONFIG_DEFAULTS = {
    "pca": dict(k_dim=500, layers=1, activation="softplus"),
    "dm":  dict(k_dim=64,  layers=4, activation="softplus"),
}

# ── shared hyperparameters (same for both configs) ─────────────────────────
SHARED_DEFAULTS = dict(
    loss="euclidean",
    pretrain_epochs=500,
    train_epochs=2500,
    train_lr=0.01,
    train_dt=0.1,
    train_sd=0.5,
    train_tau=1e-6,
    train_batch=0.1,
    train_clip=0.25,
    save=100,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a PRESCIENT model (calls prescient.commands.train_model in-process)"
    )
    p.add_argument(
        "--data_path",
        required=True,
        help="Path to processed data.pt (output of 01_prepare_data.py)",
    )
    p.add_argument(
        "--out_dir",
        required=True,
        help="Output directory (model saved to out_dir/weight_name-activation_layers_kdim-tau/seed_SEED/)",
    )
    p.add_argument(
        "--config",
        choices=["pca", "dm"],
        default=None,
        help="Preset config: 'pca' → k_dim=500, layers=1; 'dm' → k_dim=64, layers=4",
    )
    p.add_argument(
        "--weight_name",
        default="growth_weights",
        help="Descriptive weight name used in output directory name (default: derived from --config)",
    )
    # Network architecture (override preset)
    p.add_argument("--k_dim", type=int, default=None, help="Hidden units (default: 500 for pca, 64 for dm)")
    p.add_argument("--layers", type=int, default=None, help="Network depth (default: 1 for pca, 4 for dm)")
    p.add_argument("--activation", default=None, help="Activation function (default: softplus)")
    # Training hyperparams (override shared defaults)
    p.add_argument("--train_tau",        type=float, default=None, help="Tau (default: 1e-6)")
    p.add_argument("--train_dt",         type=float, default=None, help="Simulation timestep (default: 0.1)")
    p.add_argument("--train_sd",         type=float, default=None, help="Gaussian noise std dev (default: 0.5)")
    p.add_argument("--pretrain_epochs",  type=int,   default=None, help="Pre-train epochs (default: 500)")
    p.add_argument("--train_epochs",     type=int,   default=None, help="Training epochs (default: 2500)")
    p.add_argument("--train_lr",         type=float, default=None, help="Training learning rate (default: 0.01)")
    p.add_argument("--train_batch",      type=float, default=None, help="Batch size fraction (default: 0.1)")
    p.add_argument("--train_clip",       type=float, default=None, help="Gradient clip threshold (default: 0.25)")
    p.add_argument("--save",             type=int,   default=None, help="Checkpoint save frequency in epochs (default: 100)")
    # Execution
    p.add_argument("--seed", type=int, default=2,     help="Random seed (default: 2)")
    p.add_argument("--gpu",  type=int, default=None,  help="CUDA device index, e.g. 0 (default: CPU)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── resolve hyperparameters ────────────────────────────────────────────
    # Start from shared defaults, layer in config preset, then CLI overrides
    hp = dict(SHARED_DEFAULTS)

    if args.config is not None:
        hp.update(CONFIG_DEFAULTS[args.config])

    # CLI overrides
    if args.k_dim        is not None: hp["k_dim"]          = args.k_dim
    if args.layers       is not None: hp["layers"]          = args.layers
    if args.activation   is not None: hp["activation"]      = args.activation
    if args.train_tau    is not None: hp["train_tau"]       = args.train_tau
    if args.train_dt     is not None: hp["train_dt"]        = args.train_dt
    if args.train_sd     is not None: hp["train_sd"]        = args.train_sd
    if args.pretrain_epochs is not None: hp["pretrain_epochs"] = args.pretrain_epochs
    if args.train_epochs is not None: hp["train_epochs"]    = args.train_epochs
    if args.train_lr     is not None: hp["train_lr"]        = args.train_lr
    if args.train_batch  is not None: hp["train_batch"]     = args.train_batch
    if args.train_clip   is not None: hp["train_clip"]      = args.train_clip
    if args.save         is not None: hp["save"]            = args.save

    # Ensure required keys have values even without --config
    for key in ("k_dim", "layers", "activation"):
        if key not in hp:
            raise ValueError(
                f"--{key} must be set (or use --config pca/dm to apply a preset)"
            )

    # ── weight_name ───────────────────────────────────────────────────────
    weight_name = args.weight_name
    if weight_name is None:
        if args.config:
            weight_name = args.config.upper()
        else:
            weight_name = f"custom_{hp['k_dim']}_{hp['layers']}"

    os.makedirs(args.out_dir, exist_ok=True)

    # ── log resolved hyperparameters ─────────────────────────────────────
    log.info("Hyperparameters resolved:")
    for k, v in hp.items():
        log.info(f"  {k} = {v}")
    log.info(f"weight_name = {weight_name}")
    log.info(f"seed        = {args.seed}")
    log.info(f"gpu         = {args.gpu}")

    # ── build PRESCIENT args namespace ───────────────────────────────────
    # Call PRESCIENT as a Python import (not subprocess) so torch.load/save
    # monkey-patches take effect throughout training.
    from argparse import Namespace
    from prescient.commands.train_model import main as prescient_train, train_init

    prescient_args = Namespace(
        data_path=args.data_path,
        out_dir=args.out_dir,
        weight_name=weight_name,
        activation=str(hp["activation"]),
        k_dim=int(hp["k_dim"]),
        layers=int(hp["layers"]),
        loss=str(hp.get("loss", "euclidean")),
        pretrain_epochs=int(hp["pretrain_epochs"]),
        train_epochs=int(hp["train_epochs"]),
        train_lr=float(hp["train_lr"]),
        train_dt=float(hp["train_dt"]),
        train_sd=float(hp["train_sd"]),
        train_tau=float(hp["train_tau"]),
        train_batch=float(hp["train_batch"]),
        train_clip=float(hp["train_clip"]),
        save=int(hp["save"]),
        seed=args.seed,
        gpu=args.gpu if args.gpu is not None else 7,
        no_cuda=False,
        pretrain=True,
        train=True,
        config=None,
    )

    log.info(f"\nCalling prescient.commands.train_model.main() in-process ...")
    prescient_train(prescient_args)

    # ── report expected output location ───────────────────────────────────
    # PRESCIENT saves to: out_dir/weight_name-activation_layers_kdim-tau/seed_SEED/
    expected_subdir = (
        f"{weight_name}-{hp['activation']}_{hp['layers']}_{hp['k_dim']}-{hp['train_tau']}"
        f"/seed_{args.seed}"
    )
    expected_path = os.path.join(args.out_dir, expected_subdir)
    log.info(f"\nTraining complete. Expected model location:")
    log.info(f"  {expected_path}")
    log.info(f"  train.best.pt  ←  best checkpoint")
    log.info(f"  config.pt      ←  training configuration")


if __name__ == "__main__":
    main()
