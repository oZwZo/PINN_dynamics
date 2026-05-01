#!/usr/bin/env python
"""
klein_fate_eval_scdiffeq.py
===========================

Evaluate a trained scDiffEq checkpoint on Klein with the SAME metrics and
cell selections used for PRESCIENT / OT-CFM / SF2M, via the shared
``scripts/fate_eval_pipeline.py`` utilities.

Outputs (per checkpoint, under ``{output_dir}/fate_prediction_metrics/{ckpt_name}/``):

    eval_combined.csv          one row: sim_mode, accuracy, pearson_r,
                               w2_scaled, w2_raw, n_start_cells, n_sims_fate, k
    F_hat_{mode}.csv           per-start-cell predicted fate proportions
    w2_per_clone_{mode}.csv    per-clone W2 distances

``mode`` is ``"ode"`` for LightningODE-based checkpoints and ``"sde"`` otherwise.

Alignment with prescient / otcfm / sf2m
---------------------------------------
* Start cells = day-first cells in ``adata`` that also appear in ``F_obs.index``
  (NO Well filter — start cells may lie in the training set).
* kNN reference = ALL train cells (annotations from ``adata.obs[annotation_col]``).
* W2 source / target = Well==2 cells whose clone ∈ ``clone_proportions.index``,
  at ``tp_mid`` and ``tp_last`` respectively.
* scaler = None (scdiffeq trains on raw X_pca_30 / DM_EigenVectors → w2_raw == w2_scaled).
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

# Make fate_eval_pipeline importable (scripts/ is one level up from scripts/scdiffeq/)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import fate_eval_pipeline as fate_eval  # noqa: E402


CONFIGS = {
    "pca30": {
        "use_key": "X_pca_30",
        "latent_dim": 30,
        "truncate_pca": 30,
    },
    "dm10": {
        "use_key": "DM_EigenVectors",
        "latent_dim": 10,
    },
}


# --- CLI -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="scDiffEq fate + W2 evaluation on Klein "
                    "(shared pipeline with prescient/otcfm/sf2m)"
    )
    p.add_argument("--config", required=True, choices=list(CONFIGS))
    p.add_argument("--data",
                   default="/home/wz369/rds/hpc-work/PINN_dynamics/data/klein_addpop.h5ad")
    p.add_argument("--ckpt_path", required=True, type=pathlib.Path)
    p.add_argument("--f_obs",
                   default="/home/wz369/rds/hpc-work/PINN_dynamics/data/klein/F_obs.csv")
    p.add_argument("--clone_proportions",
                   default="/home/wz369/rds/hpc-work/PINN_dynamics/data/klein/clone_proportions.csv",
                   help="Clone proportions CSV — clones with index ∈ this file are used for W2.")
    p.add_argument("--output_dir", required=True, type=pathlib.Path,
                   help="Parent dir — actual outputs land in "
                        "{output_dir}/fate_prediction_metrics/{ckpt_name}/")
    p.add_argument("--ckpt_name", default=None,
                   help="Subdir name under fate_prediction_metrics/. "
                        "Defaults to ckpt stem (normalized like VersionAccuracy).")
    p.add_argument("--N", type=int, default=200,
                   help="Trajectories per start cell for SDE fate accuracy")
    p.add_argument("--n_sims_w2", type=int, default=10,
                   help="Trajectories per cell for SDE W2 (matches sf2m default)")
    p.add_argument("--k", type=int, default=15,
                   help="kNN neighbours for cell-state annotation")
    p.add_argument("--time_key", default="timepoint_tx_days")
    p.add_argument("--annotation_col", default="Annotation")
    p.add_argument("--clone_col", default="clones")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the train/val/test split (must match training)")
    p.add_argument("--device", default=None,
                   help="Defaults to autodevice.AutoDevice()")
    return p.parse_args()


# --- data prep -------------------------------------------------------------

def load_klein(data_path: pathlib.Path, config_name: str) -> ad.AnnData:
    adata = ad.read_h5ad(data_path)
    cfg = CONFIGS[config_name]
    if "truncate_pca" in cfg:
        n_pcs = cfg["truncate_pca"]
        adata.obsm[cfg["use_key"]] = adata.obsm["X_pca"][:, :n_pcs].astype(np.float32)
        print(f"Truncated X_pca → {cfg['use_key']} ({n_pcs} dims)")
    print(f"Loaded {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
    return adata


def recreate_splits(adata: ad.AnnData, seed: int) -> None:
    """Match run_scdiffeq.py / evaluate_scdiffeq.py split exactly."""
    rng = np.random.default_rng(seed)
    test_mask = (adata.obs["Well"] == 2).values
    non_test_idx = np.where(~test_mask)[0]
    n_val = int(len(non_test_idx) * 0.1)
    val_pos = rng.choice(len(non_test_idx), size=n_val, replace=False)
    val_mask_subset = np.zeros(len(non_test_idx), dtype=bool)
    val_mask_subset[val_pos] = True

    train_col = np.zeros(adata.n_obs, dtype=bool)
    val_col = np.zeros(adata.n_obs, dtype=bool)
    test_col = np.zeros(adata.n_obs, dtype=bool)
    train_col[non_test_idx[~val_mask_subset]] = True
    val_col[non_test_idx[val_mask_subset]] = True
    test_col[test_mask] = True

    adata.obs["train"] = train_col
    adata.obs["val"] = val_col
    adata.obs["test"] = test_col
    print(f"Splits — train: {train_col.sum():,}  val: {val_col.sum():,}  "
          f"test: {test_col.sum():,}")


# --- scdiffeq simulation_func factory --------------------------------------

def _make_scdiffeq_sim_fn(use_key: str, t_start: float, t_end: float, dt: float):
    """Adapter that makes ``sdq.tl.simulate`` conform to fate_eval_pipeline's contract:

        sim_fn(start_cells, model, n_sims, device) -> ndarray
            (n_cells, n_dims)            if n_sims == 1
            (n_cells, n_sims, n_dims)    if n_sims > 1

    Each call wraps ``start_cells`` in a throwaway AnnData keyed by ``use_key``
    so arbitrary embedding matrices can be consumed. The integration window is
    set explicitly via ``t=linspace(t_start, t_end, n_steps)``, which bypasses
    the AnnData's ``time_key`` (per ``Simulation._T_GIVEN``).

    Endpoint layout from scdiffeq at ``t == t_max`` is ``[sim_i outer, cell_idx inner]``
    (see Simulation._to_adata_sim), so ``reshape(n_sims, n, d).transpose(1, 0, 2)``
    yields the ``(n_cells, n_sims, n_dims)`` shape the pipeline expects.
    """
    n_steps = int(round((t_end - t_start) / dt)) + 1

    def sim_fn(start_cells, model, n_sims, device):
        n = int(start_cells.shape[0])
        tmp = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
        tmp.obs_names = [f"tmp_{i}" for i in range(n)]
        tmp.obs["_t"] = 0.0  # unused — overridden by the `t` arg below
        tmp.obsm[use_key] = np.asarray(start_cells, dtype=np.float32)

        t_tensor = torch.linspace(t_start, t_end, n_steps).to(device)

        # Plain (Lightning{ODE,SDE}) diffeqs don't need autograd inside
        # .forward(); without torch.no_grad() the retained SDE backward graph
        # OOMs an 80 GB A100 at N=200 × 2031 cells. FixedPotential models
        # however compute their drift as ∇_y ψ(y) via torch.autograd.grad
        # with create_graph=True, so no_grad would make ψ have no grad_fn —
        # we leave autograd enabled for those.
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
        endpoints = X_final.reshape(n_sims, n, -1).transpose(1, 0, 2)  # (n, n_sims, d)
        return endpoints[:, 0, :] if n_sims == 1 else endpoints

    return sim_fn


# --- misc ------------------------------------------------------------------

def _format_ckpt_name(ckpt_path: pathlib.Path) -> str:
    """Mirror VersionAccuracy._format_ckpt_name: strip suffix, '=' → '_', '-' → '.'."""
    return ckpt_path.stem.replace("=", "_").replace("-", ".")


# --- main ------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    device = args.device
    if device is None:
        device = autodevice.AutoDevice()
    else:
        device = torch.device(device)

    cfg = CONFIGS[args.config]
    ckpt_name = args.ckpt_name or _format_ckpt_name(args.ckpt_path)

    out_root = pathlib.Path(args.output_dir) / "fate_prediction_metrics" / ckpt_name
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"scDiffEq eval — config: {args.config}")
    print(f"ckpt : {args.ckpt_path}")
    print(f"out  : {out_root}")
    print("=" * 60)

    # 1. Data + splits + references
    adata = load_klein(args.data, args.config)
    recreate_splits(adata, args.seed)

    F_obs = pd.read_csv(args.f_obs, index_col=0)
    F_obs.index = F_obs.index.astype(str)
    clone_proportions = pd.read_csv(args.clone_proportions, index_col=0)
    clone_set = set(clone_proportions.index.astype(str))
    print(f"F_obs: {F_obs.shape[0]} rows × {F_obs.shape[1]} fates   "
          f"clone_proportions: {len(clone_set)} clones")

    # 2. DiffEq + inferred mode + dt
    print(f"Loading DiffEq from {args.ckpt_path} ...")
    diffeq = sdq.io.load_diffeq(ckpt_path=args.ckpt_path)
    diffeq.to(device)
    diffeq.eval()
    dt = float(diffeq.hparams["dt"])
    mode = "ode" if "ODE" in type(diffeq).__name__ else "sde"

    # FixedPotential SDEs compute drift via torch.autograd.grad(..., create_graph=True).
    # That's needed for training (to backprop through ∇ψ) but in a forward-only eval
    # it retains the SDE graph across all 41 timesteps × N sims × n cells and OOMs an
    # 80 GB A100. For eval we only need first-order drift gradients, so patch
    # create_graph → False on the PotentialSDE class.
    if "FixedPotential" in type(diffeq).__name__:
        from neural_diffeqs import PotentialSDE

        def _eval_gradient(self, ψ, y):
            return torch.autograd.grad(ψ, y, torch.ones_like(ψ), create_graph=False)[0]

        PotentialSDE._gradient = _eval_gradient
        print("Patched PotentialSDE._gradient: create_graph=False for eval")

    n_sims_fate = 1 if mode == "ode" else args.N
    n_sims_w2 = 1 if mode == "ode" else args.n_sims_w2
    print(f"DiffEq type: {type(diffeq).__name__}  mode={mode}  "
          f"device: {device}  dt={dt}  n_sims_fate={n_sims_fate}  "
          f"n_sims_w2={n_sims_w2}")

    # 3. Start cells for fate: day-first ∩ F_obs.index (full adata, no Well filter)
    tp_values = sorted(adata.obs[args.time_key].unique())
    tp_first, tp_mid, tp_last = float(tp_values[0]), float(tp_values[1]), float(tp_values[-1])
    fobs_idx_set = set(F_obs.index)
    valid_mask = adata.obs_names.astype(str).isin(fobs_idx_set)
    start_mask = valid_mask & (adata.obs[args.time_key] == tp_first).values
    start_adata = adata[start_mask]
    start_cells = np.asarray(
        start_adata.obsm[cfg["use_key"]][:, :cfg["latent_dim"]], dtype=np.float32
    )
    start_ids = np.asarray(start_adata.obs_names.astype(str))
    print(f"Fate start cells: {len(start_cells)} (day {tp_first} ∩ F_obs)")

    # 4. kNN reference = all train cells
    train_ad = adata[adata.obs["train"].values]
    x_ref = np.asarray(
        train_ad.obsm[cfg["use_key"]][:, :cfg["latent_dim"]], dtype=np.float32
    )
    y_ref = train_ad.obs[args.annotation_col].values.astype(str)
    print(f"kNN ref: {len(x_ref)} train cells (labels from '{args.annotation_col}')")

    # 5. W2 src / tgt: Well==2 ∩ clone ∈ clone_proportions, at tp_mid / tp_last
    test_mask = (adata.obs["Well"] == 2).values
    clone_mask = adata.obs[args.clone_col].astype(str).isin(clone_set).values
    w2_mask = test_mask & clone_mask
    src_mask = w2_mask & (adata.obs[args.time_key] == tp_mid).values
    tgt_mask = w2_mask & (adata.obs[args.time_key] == tp_last).values

    src_ad = adata[src_mask]
    tgt_ad = adata[tgt_mask]
    src_cells_w2 = np.asarray(
        src_ad.obsm[cfg["use_key"]][:, :cfg["latent_dim"]], dtype=np.float32
    )
    tgt_cells_w2 = np.asarray(
        tgt_ad.obsm[cfg["use_key"]][:, :cfg["latent_dim"]], dtype=np.float32
    )
    src_clones_w2 = src_ad.obs[args.clone_col].astype(str).values
    tgt_clones_w2 = tgt_ad.obs[args.clone_col].astype(str).values
    print(f"W2 src (day {tp_mid}): {len(src_cells_w2)}   "
          f"tgt (day {tp_last}): {len(tgt_cells_w2)}")

    # 6. Build sim_fns
    sim_fn_fate = _make_scdiffeq_sim_fn(cfg["use_key"], tp_first, tp_last, dt)
    sim_fn_w2 = _make_scdiffeq_sim_fn(cfg["use_key"], tp_mid, tp_last, dt)

    # 7. Fate evaluation (accuracy + Pearson r)
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

    # 8. Population W2 + per-clone W2 (scaler=None — raw space)
    print("\n--- POPULATION W2 ---")
    w2_pop = fate_eval.compute_w2_population(
        src_cells=src_cells_w2,
        target_cells=tgt_cells_w2,
        model=diffeq,
        sim_fn=sim_fn_w2,
        n_sims=n_sims_w2,
        device=device,
        scaler=None,
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
        scaler=None,
    )
    w2_clone.to_csv(out_root / f"w2_per_clone_{mode}.csv", index=False)
    print(f"  per-clone W2: {len(w2_clone)} clones")

    # 9. eval_combined.csv — identical schema to prescient/otcfm/sf2m
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
    for p in sorted(out_root.iterdir()):
        print(f"  {p}")
    print("=" * 60)


if __name__ == "__main__":
    main()
