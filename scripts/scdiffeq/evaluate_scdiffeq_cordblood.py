#!/usr/bin/env python
"""
evaluate_scdiffeq_cordblood.py
===============================

Evaluate a trained scDiffEq checkpoint on cord-blood (CLADES) using the SAME
metrics and cell-selections as PRESCIENT / OT-CFM / SF2M, via the shared
``scripts/fate_eval_pipeline.py`` utilities.

Cord-blood specifics vs Klein:
  - obs split column  : ``split`` ∈ {train, val, test}  (not integer ``Well``)
  - test cells        : split == "test"  (mirrored in Well==2 by preprocessing)
  - timepoint column  : ``timepoint_tx_days`` (int: 3, 10, 17)
  - cell-type column  : ``def_lab``
  - embeddings        : ``X_pca_scaled`` (30d) / ``DM_EigenVectors_scaled`` (9d)
  - F_obs             : data/CordBlood/F_obs_day17.csv  (9073 × 12)
  - clone_proportions : data/CordBlood/clone_proportions_day17.csv
  - scaler            : read from adata.uns['PC_scaler'] / adata.uns['DM_scaler']

Outputs (under ``{output_dir}/fate_prediction_metrics/{ckpt_name}/``):
    eval_combined.csv
    F_hat_{mode}.csv
    w2_per_clone_{mode}.csv

Usage:
    uv run --directory /rds/user/wz369/hpc-work/scDiffEq python \\
        /rds/user/wz369/hpc-work/PINN_dynamics/scripts/scdiffeq/evaluate_scdiffeq_cordblood.py \\
        --config pca30 \\
        --ckpt_path logs/cordblood/scdiffeq/pca30_fp_vr_s42/lightning_logs/version_0/checkpoints/epoch=99-step=4700.ckpt \\
        --output_dir logs/cordblood/scdiffeq/pca30_fp_vr_s42/eval
"""

import argparse
import os
import pathlib
import sys

import anndata as ad
import autodevice
import numpy as np
import pandas as pd
import scdiffeq as sdq
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import fate_eval_pipeline as fate_eval  # noqa: E402

# ---------------------------------------------------------------------------
# Embedding configs (must match run_scdiffeq_cordblood.py)
# ---------------------------------------------------------------------------
CONFIGS = {
    "pca30": {
        "use_key": "X_pca_scaled",
        "latent_dim": 30,
    },
    "dm9": {
        "use_key": "DM_EigenVectors_scaled",
        "latent_dim": 9,
    },
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="scDiffEq fate + W2 evaluation on cord-blood "
                    "(shared pipeline with prescient/otcfm/sf2m)"
    )
    p.add_argument("--config", required=True, choices=list(CONFIGS))
    p.add_argument(
        "--data",
        default="/rds/user/wz369/hpc-work/PINN_dynamics/data/cordblood_addpop.h5ad",
    )
    p.add_argument("--ckpt_path", required=True, type=pathlib.Path)
    p.add_argument(
        "--f_obs",
        default="/rds/user/wz369/hpc-work/PINN_dynamics/data/CordBlood/F_obs_day17.csv",
    )
    p.add_argument(
        "--clone_proportions",
        default="/rds/user/wz369/hpc-work/PINN_dynamics/data/CordBlood/clone_proportions_day17.csv",
        help="Clone proportions CSV — clones in this file are used for W2.",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        type=pathlib.Path,
        help="Outputs land in {output_dir}/fate_prediction_metrics/{ckpt_name}/",
    )
    p.add_argument(
        "--ckpt_name",
        default=None,
        help="Subdir under fate_prediction_metrics/. Defaults to ckpt stem.",
    )
    p.add_argument("--N", type=int, default=200,
                   help="Trajectories per start cell for SDE fate accuracy")
    p.add_argument("--n_sims_w2", type=int, default=10,
                   help="Trajectories per cell for SDE W2")
    p.add_argument("--k", type=int, default=15, help="kNN neighbours")
    p.add_argument("--time_key", default="timepoint_tx_days")
    p.add_argument("--celltype_col", default="def_lab")
    p.add_argument("--clone_col", default="clones")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed — must match the training seed for split reproducibility")
    p.add_argument("--device", default=None,
                   help="Defaults to autodevice.AutoDevice()")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Simulation adapter (same as Klein version)
# ---------------------------------------------------------------------------

def _make_scdiffeq_sim_fn(use_key: str, t_start: float, t_end: float, dt: float):
    """Adapter so sdq.tl.simulate conforms to fate_eval_pipeline's contract."""
    n_steps = int(round((t_end - t_start) / dt)) + 1

    def sim_fn(start_cells, model, n_sims, device):
        n = int(start_cells.shape[0])
        tmp = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
        tmp.obs_names = [f"tmp_{i}" for i in range(n)]
        tmp.obs["_t"] = 0.0
        tmp.obsm[use_key] = np.asarray(start_cells, dtype=np.float32)
        t_tensor = torch.linspace(t_start, t_end, n_steps).to(device)
        needs_grad = "FixedPotential" in type(model).__name__
        ctx = torch.enable_grad() if needs_grad else torch.no_grad()
        with ctx:
            adata_sim = sdq.tl.simulate(
                adata=tmp,
                diffeq=model,
                idx=pd.Index(tmp.obs_names),
                use_key=use_key,
                time_key="_t",
                N=n_sims,
                dt=dt,
                t=t_tensor,
                device=device,
            )
        final_mask = (adata_sim.obs["t"] == adata_sim.obs["t"].max()).values
        X_final = np.asarray(adata_sim.X[final_mask], dtype=np.float32)
        endpoints = X_final.reshape(n_sims, n, -1).transpose(1, 0, 2)
        return endpoints[:, 0, :] if n_sims == 1 else endpoints

    return sim_fn


def _format_ckpt_name(ckpt_path: pathlib.Path) -> str:
    return ckpt_path.stem.replace("=", "_").replace("-", ".")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    device = args.device
    if device is None:
        device = autodevice.AutoDevice()
    else:
        device = torch.device(device)

    cfg = CONFIGS[args.config]
    use_key = cfg["use_key"]
    latent_dim = cfg["latent_dim"]
    ckpt_name = args.ckpt_name or _format_ckpt_name(args.ckpt_path)

    out_root = pathlib.Path(args.output_dir) / "fate_prediction_metrics" / ckpt_name
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"scDiffEq cord-blood eval — config: {args.config}")
    print(f"ckpt : {args.ckpt_path}")
    print(f"out  : {out_root}")
    print("=" * 60)

    # 1. Data — keep original barcode obs_names so F_obs/clone lookups work.
    # (The training script resets obs_names to 0-based integers for scDiffEq's
    # internal dataloader; the eval script works directly with barcodes.)
    adata = ad.read_h5ad(args.data)
    adata.obs_names_make_unique()
    # Do NOT reset obs index — barcodes must match F_obs.index
    print(f"Loaded {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

    # Recreate train/val/test markers (needed for kNN reference selection)
    if "split" in adata.obs.columns:
        adata.obs["train"] = (adata.obs["split"] == "train").values
        adata.obs["val"]   = (adata.obs["split"] == "val").values
        adata.obs["test"]  = (adata.obs["split"] == "test").values
    else:
        # Fallback: Well integer
        rng = np.random.default_rng(args.seed)
        test_mask = (adata.obs["Well"] == 2).values
        non_test_idx = np.where(~test_mask)[0]
        n_val = int(len(non_test_idx) * 0.1)
        val_pos = rng.choice(len(non_test_idx), size=n_val, replace=False)
        vm = np.zeros(len(non_test_idx), dtype=bool)
        vm[val_pos] = True
        train_col = np.zeros(adata.n_obs, dtype=bool)
        val_col   = np.zeros(adata.n_obs, dtype=bool)
        train_col[non_test_idx[~vm]] = True
        val_col[non_test_idx[vm]] = True
        adata.obs["train"] = train_col
        adata.obs["val"]   = val_col
        adata.obs["test"]  = test_mask

    F_obs = pd.read_csv(args.f_obs, index_col=0)
    F_obs.index = F_obs.index.astype(str)
    clone_proportions = pd.read_csv(args.clone_proportions, index_col=0)
    clone_set = set(clone_proportions.index.astype(str))
    print(f"F_obs: {F_obs.shape[0]} rows × {F_obs.shape[1]} fates   "
          f"clone_proportions: {len(clone_set)} clones")

    # 2. DiffEq + mode + dt
    print(f"Loading DiffEq from {args.ckpt_path} ...")
    diffeq = sdq.io.load_diffeq(ckpt_path=args.ckpt_path)
    diffeq.to(device)
    diffeq.eval()
    dt = float(diffeq.hparams["dt"])
    mode = "ode" if "ODE" in type(diffeq).__name__ else "sde"

    if "FixedPotential" in type(diffeq).__name__:
        from neural_diffeqs import PotentialSDE

        def _eval_gradient(self, psi, y):
            return torch.autograd.grad(psi, y, torch.ones_like(psi), create_graph=False)[0]

        PotentialSDE._gradient = _eval_gradient
        print("Patched PotentialSDE._gradient: create_graph=False for eval")

    n_sims_fate = 1 if mode == "ode" else args.N
    n_sims_w2   = 1 if mode == "ode" else args.n_sims_w2
    print(f"DiffEq: {type(diffeq).__name__}  mode={mode}  dt={dt}  "
          f"n_sims_fate={n_sims_fate}  n_sims_w2={n_sims_w2}")

    # 3. Scaler (cord-blood embeddings are pre-scaled; read from adata.uns)
    scaler = fate_eval.compute_scaler_from_adata(adata, use_key, latent_dim)
    if scaler is not None:
        print(f"Scaler loaded from adata.uns for '{use_key}': mean shape={scaler['mean'].shape}")
    else:
        print("Scaler: None (w2_raw == w2_scaled)")

    # 4. Start cells for fate (day-first ∩ F_obs.index, all cells / no test filter)
    tp_values = sorted(adata.obs[args.time_key].unique())
    tp_first, tp_mid, tp_last = float(tp_values[0]), float(tp_values[1]), float(tp_values[-1])
    fobs_set = set(F_obs.index)
    valid_mask = adata.obs_names.astype(str).isin(fobs_set)
    start_mask = valid_mask & (adata.obs[args.time_key] == tp_first).values
    start_adata = adata[start_mask]
    start_cells = np.asarray(
        start_adata.obsm[use_key][:, :latent_dim], dtype=np.float32
    )
    start_ids = np.asarray(start_adata.obs_names.astype(str))
    print(f"Fate start cells: {len(start_cells)} (day {tp_first} ∩ F_obs)")

    # 5. kNN reference = all train cells
    train_ad = adata[adata.obs["train"].values]
    x_ref = np.asarray(train_ad.obsm[use_key][:, :latent_dim], dtype=np.float32)
    y_ref = train_ad.obs[args.celltype_col].values.astype(str)
    print(f"kNN ref: {len(x_ref)} train cells ('{args.celltype_col}')")

    # 6. W2 src / tgt: test cells ∩ clone ∈ clone_proportions, at tp_mid / tp_last
    # Cord blood uses split column; mirror Well==2 convention for fate_eval_pipeline
    test_mask = adata.obs["test"].values
    clone_mask = adata.obs[args.clone_col].astype(str).isin(clone_set).values
    w2_mask = test_mask & clone_mask
    src_mask = w2_mask & (adata.obs[args.time_key] == tp_mid).values
    tgt_mask = w2_mask & (adata.obs[args.time_key] == tp_last).values
    src_ad = adata[src_mask]
    tgt_ad = adata[tgt_mask]
    src_cells_w2 = np.asarray(src_ad.obsm[use_key][:, :latent_dim], dtype=np.float32)
    tgt_cells_w2 = np.asarray(tgt_ad.obsm[use_key][:, :latent_dim], dtype=np.float32)
    src_clones_w2 = src_ad.obs[args.clone_col].astype(str).values
    tgt_clones_w2 = tgt_ad.obs[args.clone_col].astype(str).values
    print(f"W2 src (day {tp_mid}): {len(src_cells_w2)}   "
          f"tgt (day {tp_last}): {len(tgt_cells_w2)}")

    # 7. Build sim functions
    sim_fn_fate = _make_scdiffeq_sim_fn(use_key, tp_first, tp_last, dt)
    sim_fn_w2   = _make_scdiffeq_sim_fn(use_key, tp_mid,  tp_last, dt)

    # 8. Fate evaluation
    print("\n--- FATE EVALUATION ---")
    fate_result = fate_eval.run_fate_evaluation(
        start_cells=start_cells,
        start_cell_ids=start_ids,
        model=diffeq,
        simulation_func=sim_fn_fate,
        F_obs=F_obs,
        x_ref=x_ref,
        y_ref=y_ref,
        n_sims=n_sims_fate,
        k=args.k,
        device=device,
    )
    print(f"  accuracy  : {fate_result['accuracy']:.4f}")
    print(f"  pearson_r : {fate_result['pearson_r']:.4f}")
    fate_result["F_hat"].to_csv(out_root / f"F_hat_{mode}.csv")

    # 9. Population W2 + per-clone W2
    print("\n--- POPULATION W2 ---")
    w2_pop = fate_eval.compute_w2_population(
        src_cells=src_cells_w2,
        target_cells=tgt_cells_w2,
        model=diffeq,
        sim_fn=sim_fn_w2,
        n_sims=n_sims_w2,
        device=device,
        scaler=scaler,
    )
    print(f"  w2_scaled : {w2_pop['w2_scaled']:.4f}")
    print(f"  w2_raw    : {w2_pop['w2_raw']:.4f}")

    print("\n--- PER-CLONE W2 ---")
    w2_clone = fate_eval.compute_w2_per_clone(
        src_cells=src_cells_w2,
        src_clone_ids=src_clones_w2,
        target_cells=tgt_cells_w2,
        target_clone_ids=tgt_clones_w2,
        model=diffeq,
        sim_fn=sim_fn_w2,
        n_sims=n_sims_w2,
        device=device,
        scaler=scaler,
    )
    w2_clone.to_csv(out_root / f"w2_per_clone_{mode}.csv", index=False)
    print(f"  per-clone W2: {len(w2_clone)} clones")

    # 10. eval_combined.csv — identical schema to prescient/otcfm/sf2m
    pd.DataFrame([{
        "sim_mode": mode,
        "accuracy": fate_result["accuracy"],
        "pearson_r": fate_result["pearson_r"],
        "w2_scaled": w2_pop["w2_scaled"],
        "w2_raw": w2_pop["w2_raw"],
        "n_start_cells": fate_result["n_start_cells"],
        "n_sims_fate": n_sims_fate,
        "k": args.k,
    }]).to_csv(out_root / "eval_combined.csv", index=False)

    print("\n" + "=" * 60)
    print("Done. Outputs:")
    for pf in sorted(out_root.iterdir()):
        print(f"  {pf}")
    print("=" * 60)


if __name__ == "__main__":
    main()
