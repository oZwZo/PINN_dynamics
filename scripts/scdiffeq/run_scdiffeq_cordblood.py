#!/usr/bin/env python
"""Training script for scDiffEq on cord-blood (CLADES / LARRY) dataset.

Mirrors run_scdiffeq.py (Klein) but targets:
  - data/cordblood_addpop.h5ad
  - obs columns: Well / def_lab / timepoint_tx_days
  - embeddings: X_pca_scaled (30d) and DM_EigenVectors_scaled (9d)

Usage:
    uv run --directory /rds/user/wz369/hpc-work/scDiffEq python \\
        /rds/user/wz369/hpc-work/PINN_dynamics/scripts/scdiffeq/run_scdiffeq_cordblood.py \\
        --config {pca30|dm9} [--output_dir PATH] [--epochs N] [--seed INT]
"""

import argparse
import os
import sys
import numpy as np
import anndata as ad

# ---------------------------------------------------------------------------
# Cord-blood embedding configurations
# ---------------------------------------------------------------------------
CONFIGS = {
    "pca30": {
        "use_key": "X_pca_scaled",   # already z-scored (fits 30 dims directly)
        "latent_dim": 30,
    },
    "dm9": {
        "use_key": "DM_EigenVectors_scaled",  # 9 Palantir DM components, z-scored
        "latent_dim": 9,
    },
}

# Shared hyperparameters — same as Klein reference
SHARED_PARAMS = {
    "time_key": "timepoint_tx_days",
    "mu_hidden": [512, 512],
    "sigma_hidden": [32, 32],
    "mu_dropout": 0.0,
    "sigma_dropout": 0.0,
    "batch_size": 2048,
    "train_lr": 1e-4,
    "dt": 0.1,
}

# Model-variant presets — same as Klein reference
VARIANTS = {
    "plain_sde": {
        "DiffEq_type": "SDE",
        "potential_type": None,
        "velocity_ratio_params": None,
        "sde_type": "ito",
        "brownian_dim": 1,
    },
    "plain_ode": {
        "DiffEq_type": "ODE",
        "potential_type": None,
        "velocity_ratio_params": None,
    },
    "fp_vr": {
        "DiffEq_type": "SDE",
        "potential_type": "fixed",
        "velocity_ratio_params": {
            "target": 2.5,
            "enforce": 100,
            "method": "square",
        },
        "sde_type": "ito",
        "brownian_dim": 1,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train scDiffEq on cord-blood dataset")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=list(CONFIGS.keys()),
        help="Embedding: pca30 (X_pca_scaled, 30d) or dm9 (DM_EigenVectors_scaled, 9d)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="fp_vr",
        choices=list(VARIANTS.keys()),
        help="Model variant: plain_sde | plain_ode | fp_vr (default: fp_vr)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/rds/user/wz369/hpc-work/PINN_dynamics/data/cordblood_addpop.h5ad",
        help="Path to cordblood_addpop.h5ad",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/rds/user/wz369/hpc-work/PINN_dynamics/logs/cordblood/scdiffeq",
        help="Base output directory for checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2500,
        help="Number of training epochs (default: 2500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def create_train_val_test_split(adata, seed=42):
    """Create train/val/test split aligned with cord-blood Well column.

    - test : Well == "test"  (split column set by 01_preprocess.py)
    - train/val: Well != "test", split 90/10 within non-test cells.

    Cord blood uses obs['split'] ∈ {train, val, test} rather than Well integer.
    We map this directly so the scDiffEq dataloader sees correct splits.
    """
    if "split" in adata.obs.columns:
        # Use existing deterministic split produced by 01_preprocess.py
        train_col = (adata.obs["split"] == "train").values
        val_col = (adata.obs["split"] == "val").values
        test_col = (adata.obs["split"] == "test").values
        # Also mirror in Well so the fate_eval_pipeline's Well==2 filter works
        if "Well" not in adata.obs.columns:
            adata.obs["Well"] = 1
            adata.obs.loc[test_col, "Well"] = 2
    else:
        # Fallback: use Well column (Klein-style integer)
        rng = np.random.default_rng(seed)
        test_mask = (adata.obs["Well"] == 2).values
        non_test_idx = np.where(~test_mask)[0]
        n_val = int(len(non_test_idx) * 0.1)
        val_pos = rng.choice(len(non_test_idx), size=n_val, replace=False)
        val_mask_sub = np.zeros(len(non_test_idx), dtype=bool)
        val_mask_sub[val_pos] = True
        train_col = np.zeros(adata.n_obs, dtype=bool)
        val_col = np.zeros(adata.n_obs, dtype=bool)
        test_col = test_mask.copy()
        train_col[non_test_idx[~val_mask_sub]] = True
        val_col[non_test_idx[val_mask_sub]] = True

    adata.obs["train"] = train_col
    adata.obs["val"] = val_col
    adata.obs["test"] = test_col
    print(
        f"Split — train: {train_col.sum():,}  val: {val_col.sum():,}  "
        f"test: {test_col.sum():,}"
    )


def main():
    args = parse_args()

    working_dir = os.path.join(
        args.output_dir, f"{args.config}_{args.variant}_s{args.seed}"
    )
    os.makedirs(working_dir, exist_ok=True)

    print("=" * 60)
    print(f"scDiffEq Cord-Blood — Config: {args.config} | Variant: {args.variant}")
    print("=" * 60)
    print(f"Data:        {args.data}")
    print(f"Working dir: {working_dir}")
    print(f"Epochs:      {args.epochs}")
    print(f"Seed:        {args.seed}")
    print()

    # --- Load data ---
    print("Loading data...")
    adata = ad.read_h5ad(args.data)
    # Ensure contiguous string obs index
    adata.obs_names_make_unique()
    adata.obs = adata.obs.reset_index(drop=True)
    adata.obs.index = adata.obs.index.astype(str)
    print(f"Loaded {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

    # --- Splits ---
    print("Creating train/val/test splits...")
    create_train_val_test_split(adata, seed=args.seed)
    print()

    # --- Embedding check ---
    cfg = {**SHARED_PARAMS, **CONFIGS[args.config], **VARIANTS[args.variant]}
    use_key = cfg["use_key"]
    if use_key not in adata.obsm:
        raise KeyError(
            f"obsm key '{use_key}' not found in adata.obsm. "
            f"Available: {list(adata.obsm.keys())}"
        )
    latent_dim = cfg["latent_dim"]
    actual_dim = adata.obsm[use_key].shape[1]
    if actual_dim < latent_dim:
        raise ValueError(
            f"obsm['{use_key}'] has {actual_dim} dims < requested latent_dim={latent_dim}"
        )
    print(
        f"Embedding: {use_key}  shape=({adata.n_obs}, {actual_dim})  "
        f"using first {latent_dim} dims"
    )
    print()

    # --- Import scDiffEq ---
    import scdiffeq as sdq

    # --- Model kwargs ---
    model_kwargs = dict(
        use_key=use_key,
        latent_dim=latent_dim,
        time_key=cfg["time_key"],
        DiffEq_type=cfg["DiffEq_type"],
        potential_type=cfg["potential_type"],
        velocity_ratio_params=cfg["velocity_ratio_params"],
        mu_hidden=cfg["mu_hidden"],
        sigma_hidden=cfg["sigma_hidden"],
        mu_dropout=cfg["mu_dropout"],
        sigma_dropout=cfg["sigma_dropout"],
        batch_size=cfg["batch_size"],
        train_lr=cfg["train_lr"],
        dt=cfg["dt"],
        train_val_split=[0.9, 0.1],
        working_dir=working_dir,
        seed=args.seed,
    )
    if cfg["DiffEq_type"] == "SDE":
        model_kwargs["sde_type"] = cfg["sde_type"]
        model_kwargs["brownian_dim"] = cfg["brownian_dim"]

    print("Instantiating scDiffEq model...")
    model = sdq.scDiffEq(adata, **model_kwargs)
    print(f"Model type: {type(model.DiffEq).__name__}")
    print()

    # --- Train ---
    print(f"Starting training for {args.epochs} epochs...")
    model.fit(train_epochs=args.epochs)

    print()
    print("=" * 60)
    print("Training complete!")
    print(f"Results saved to: {working_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
