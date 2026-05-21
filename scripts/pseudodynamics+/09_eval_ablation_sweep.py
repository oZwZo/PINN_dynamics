#!/usr/bin/env python3
"""
09_eval_ablation_sweep.py — sweep evaluation for the T4 ablation registry.

For each (dataset, arm, seed) row in logs/ablation_census.csv this script:
  Stage W2     — population-level W2 between simulated endpoints (SDE) and
                 observed cells at a downstream timepoint. Klein uses the
                 same Well=2 / day 4→6 setup as Fig 3. tom_pos simulates
                 first→last training timepoint over all cells.
  Stage FIELDS — per-cell D, v, g, log_u evaluated at each observed t_k.
                 Saved to logs/ablation_eval/fields/<dataset>_<arm>_<seed>.npz.
  Stage HOLDOUT (tom_pos only) — predicted log_u(s, t_k) at held-out
                 timepoints t_k ∈ {5, 7}; W2 vs KDE of observed cells.
  Stage GROWTH (tom_pos only) — per (arm, seed, t_k):
                 [min g, max g, p5 g, p95 g] over observed cells, paired
                 with the interval log-slope g_int from observed counts.

Usage examples
--------------
# full sweep, all stages
python 09_eval_ablation_sweep.py

# only klein_OT, only the W2 stage
python 09_eval_ablation_sweep.py --datasets klein_OT --stages w2

# specific arms only
python 09_eval_ablation_sweep.py --arms baseline lambdag_0 lambdaD_10

The script is resumable — already-written CSV/NPZ rows are skipped unless
--overwrite is passed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch

import pseudodynamics as pdp

# When running inside the project's singularity container, the pseudodynamics
# package is bind-mounted to /usr/local/lib/python3.10/dist-packages/pseudodynamics,
# which makes pdp.main_dir resolve to /usr/local/lib/python3.10 (wrong). Resolve
# the project root explicitly: env var override > host path fallback.
PROJECT_ROOT = os.environ.get(
    "PDP_ROOT", "/rds/user/wz369/hpc-work/pseudodynamics_plus"
)
if not Path(PROJECT_ROOT).joinpath("scripts", "fate_eval_pipeline.py").exists():
    raise RuntimeError(
        f"PROJECT_ROOT={PROJECT_ROOT} does not contain scripts/fate_eval_pipeline.py. "
        "Set PDP_ROOT to the pseudodynamics_plus checkout."
    )
os.chdir(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "scripts"))
import fate_eval_pipeline as fate_eval  # noqa: E402

ROOT = Path(PROJECT_ROOT)
EVAL_DIR = ROOT / "logs" / "ablation_eval"
FIELDS_DIR = EVAL_DIR / "fields"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
FIELDS_DIR.mkdir(parents=True, exist_ok=True)

CENSUS = ROOT / "logs" / "ablation_census.csv"

# Per-dataset eval spec. t_start / t_end are in the dataset's *human-readable*
# normalized time (same convention used by make_pseudodynamics_sde_sim_fn).
# klein convention: norm_time='min_minus', tp index [0,1,2] -> days [2,4,6].
DATASET_SPEC = {
    "klein_OT": {
        "h5ad": "data/klein_addpop.h5ad",
        "tp_key": "timepoint_tx_days",
        "test_filter": ("Well", 2),  # use Well==2 as Fig 3 test split
        "w2_t_start_real": 4,        # day 4
        "w2_t_end_real": 6,          # day 6
        "w2_t_start_norm": 2.0,      # day 4 -> norm 2
        "w2_t_end_norm": 4.0,
    },
    "klein_nonOT": {
        "h5ad": "data/klein_addpop.h5ad",
        "tp_key": "timepoint_tx_days",
        "test_filter": ("Well", 2),
        "w2_t_start_real": 4,
        "w2_t_end_real": 6,
        "w2_t_start_norm": 2.0,
        "w2_t_end_norm": 4.0,
    },
    "tom_pos": {
        "h5ad": "data/tom_pos.h5ad",
        "tp_key": "timepoint_idx",   # filled in lazily — verify against adata
        "test_filter": None,
        "w2_t_start_real": 0,        # earliest training idx
        "w2_t_end_real": 8,          # latest training idx
        "w2_t_start_norm": 0.0,
        "w2_t_end_norm": 8.0,
        "holdout_tps": [5, 7],
    },
}

ADATA_CACHE: dict[str, sc.AnnData] = {}
SCALER_CACHE: dict[tuple[str, str, int], dict] = {}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_adata(dataset: str) -> sc.AnnData:
    if dataset not in ADATA_CACHE:
        path = DATASET_SPEC[dataset]["h5ad"]
        ADATA_CACHE[dataset] = sc.read_h5ad(path)
    return ADATA_CACHE[dataset]


def get_scaler(dataset: str, cellstate_key: str, n_dims: int) -> dict:
    key = (dataset, cellstate_key, n_dims)
    if key not in SCALER_CACHE:
        SCALER_CACHE[key] = fate_eval.compute_scaler_from_adata(
            get_adata(dataset), cellstate_key, n_dims
        )
    return SCALER_CACHE[key]


def load_model(ckpt_path: str, device: str):
    return pdp.models.pde_params.load_from_checkpoint(ckpt_path).to(device).eval()


def load_config(config_path: str) -> "pdp.ExperimentConfig":
    cfg = pdp.ExperimentConfig(config_path)
    # Force from_json to rewrite stored paths against the real project root,
    # not pdp.main_dir (which is wrong inside the singularity container).
    cfg.from_json(config_path, main_dir=str(ROOT) + "/")
    return cfg


# -----------------------------------------------------------------------------
# Stage W2
# -----------------------------------------------------------------------------
def _build_sim_fn(args, t_start_norm: float):
    """Same dispatch as 03_eval_pdp_w2.py."""
    if args.sim_fn == "ode":
        return fate_eval.make_pseudodynamics_sim_fn(
            t_start_norm=t_start_norm, t_end_norm=args.t_end_norm)
    if args.sim_fn == "sde":
        return fate_eval.make_pseudodynamics_sde_sim_fn(
            t_start_norm=t_start_norm, t_end_norm=args.t_end_norm,
            n_steps=args.n_steps, noise_scale=args.noise_scale)
    if args.sim_fn == "sb":
        return fate_eval.make_pseudodynamics_sb_sim_fn(
            t_start_norm=t_start_norm, t_end_norm=args.t_end_norm,
            n_steps=args.n_steps, noise_scale=args.noise_scale)
    raise ValueError(f"unknown sim_fn={args.sim_fn!r}")


def stage_w2_klein(row, config, model, device, args):
    """Direct call to klein_w2_v1 — same code path as 03_eval_pdp_w2.py."""
    adata = get_adata(row["dataset"])
    test_ad = adata[adata.obs.Well == 2]
    clone_proportions = pd.read_csv("data/klein/clone_proportions.csv", index_col=0)
    cellstate_key = config.dataset_config.get("cellstate_key", None)
    n_dims = config.dataset_config["n_dimension"]
    scaler = fate_eval.compute_scaler_from_adata(adata, cellstate_key, n_dims)
    sim_fn = _build_sim_fn(args, t_start_norm=2.0)

    out = []
    for rep in range(args.n_sims_replicates):
        w2 = fate_eval.klein_w2_v1(
            test_ad, clone_proportions, config, device, sim_fn,
            n_sims=args.n_sims_per_cell, scaler=scaler,
        )
        out.append({"replicate": rep + 1,
                    "w2_scaled": float(w2["w2_scaled"]),
                    "w2_raw":    float(w2["w2_raw"])})
    return out


def stage_w2_tompos(row, config, model, device, args):
    """tom_pos W2 — first → last training tp, all cells."""
    spec = DATASET_SPEC[row["dataset"]]
    adata = get_adata(row["dataset"])
    cellstate_key = config.dataset_config["cellstate_key"]
    n_dims = config.dataset_config["n_dimension"]
    scaler = fate_eval.compute_scaler_from_adata(adata, cellstate_key, n_dims)
    tp_key = spec["tp_key"] if spec["tp_key"] in adata.obs.columns else _autoselect_tp_key(adata)

    start_cells = adata[adata.obs[tp_key] == spec["w2_t_start_real"]].obsm[cellstate_key][:, :n_dims]
    ref_cells   = adata[adata.obs[tp_key] == spec["w2_t_end_real"]].obsm[cellstate_key][:, :n_dims]
    if len(start_cells) == 0 or len(ref_cells) == 0:
        return []

    sim_fn = _build_sim_fn(args, t_start_norm=float(spec["w2_t_start_real"]))
    cap = args.w2_cap_cells

    out = []
    for rep in range(args.n_sims_replicates):
        endpoints = np.asarray(sim_fn(start_cells, model, args.n_sims_per_cell, device))
        endpoints = endpoints.reshape(-1, start_cells.shape[1])
        if endpoints.shape[0] > cap:
            endpoints = endpoints[np.random.choice(endpoints.shape[0], cap, replace=False)]
        ref = ref_cells if ref_cells.shape[0] <= cap else \
              ref_cells[np.random.choice(ref_cells.shape[0], cap, replace=False)]
        w2_scaled = float(fate_eval.compute_w2(endpoints, ref, device=device))
        if scaler is not None:
            w2_raw = float(fate_eval.compute_w2(
                fate_eval.inverse_standardize(endpoints, scaler),
                fate_eval.inverse_standardize(ref, scaler),
                device=device))
        else:
            w2_raw = w2_scaled
        out.append({"replicate": rep + 1, "w2_scaled": w2_scaled, "w2_raw": w2_raw})
    return out


def _autoselect_tp_key(adata) -> str:
    for cand in ("timepoint_idx", "timepoint", "tp", "tp_idx", "time"):
        if cand in adata.obs.columns:
            return cand
    raise KeyError(f"no timepoint column in adata.obs.columns: {list(adata.obs.columns)[:20]}")


# -----------------------------------------------------------------------------
# Stage FIELDS — per-cell D, v, g, log_u at each observed t_k
# -----------------------------------------------------------------------------
def stage_fields(row: dict, config, model, device, args) -> dict:
    """Evaluate D, v, g, log_u per observed cell at each observed t_k.

    Returns a dict of np arrays ready for np.savez_compressed.
    """
    dataset = row["dataset"]
    adata = get_adata(dataset)
    cellstate_key = config.dataset_config["cellstate_key"]
    n_dims = config.dataset_config["n_dimension"]
    spec = DATASET_SPEC[dataset]
    tp_key = spec["tp_key"] if spec["tp_key"] in adata.obs.columns else _autoselect_tp_key(adata)

    tsf = float(model.time_scale_factor)
    t_obs_real = sorted(adata.obs[tp_key].unique().tolist())

    arrays: dict[str, np.ndarray] = {
        "cellstate_key": np.array([cellstate_key]),
        "tp_key": np.array([tp_key]),
        "tps_real": np.asarray(t_obs_real, dtype=float),
        "n_dims": np.array([n_dims]),
    }

    for t_real in t_obs_real:
        sub = adata[adata.obs[tp_key] == t_real]
        s_np = sub.obsm[cellstate_key][:, :n_dims].astype(np.float32)
        cell_idx = np.asarray(sub.obs_names, dtype=str)
        s = torch.from_numpy(s_np).to(device)
        # convert real_t -> norm_t same as klein convention
        t_norm = float(t_real) - float(min(t_obs_real))
        t_in = torch.full((s.shape[0], 1), t_norm * tsf, dtype=torch.float32, device=device)
        with torch.no_grad():
            D = model.D(s, t_in).cpu().numpy()
            v = model.v(s, t_in).cpu().numpy()
            g = model.g(s, t_in).cpu().numpy()
            logu = model.u(s, t_in).cpu().numpy()
        tag = f"t{t_real}"
        arrays[f"{tag}_cell_idx"] = cell_idx
        arrays[f"{tag}_s"] = s_np
        arrays[f"{tag}_D"] = D.astype(np.float32)
        arrays[f"{tag}_v"] = v.astype(np.float32)
        arrays[f"{tag}_g"] = g.astype(np.float32)
        arrays[f"{tag}_logu"] = logu.astype(np.float32)
    return arrays


# -----------------------------------------------------------------------------
# Stage HOLDOUT (tom_pos) — predict u(s, t_k) at held-out cells; W2 vs KDE
# -----------------------------------------------------------------------------
def stage_holdout(row: dict, config, model, device, args) -> list[dict]:
    """For each held-out t_k: weight observed cells by predicted u(s,t_k), then
    W2 between weighted-resampled predicted cloud and observed cells."""
    if row["dataset"] != "tom_pos":
        return []
    spec = DATASET_SPEC["tom_pos"]
    adata = get_adata("tom_pos")
    cellstate_key = config.dataset_config["cellstate_key"]
    n_dims = config.dataset_config["n_dimension"]
    tp_key = spec["tp_key"] if spec["tp_key"] in adata.obs.columns else _autoselect_tp_key(adata)
    tsf = float(model.time_scale_factor)
    t_obs_real = sorted(adata.obs[tp_key].unique().tolist())
    t_min = float(min(t_obs_real))

    out = []
    rng = np.random.default_rng(42)
    for t_real in spec["holdout_tps"]:
        sub_obs = adata[adata.obs[tp_key] == t_real]
        if sub_obs.n_obs == 0:
            continue
        # Observed cells at the held-out timepoint (ground truth cloud)
        s_obs = sub_obs.obsm[cellstate_key][:, :n_dims].astype(np.float32)

        # Predicted cloud: take ALL cells (across all tps), weight by exp(log_u(s, t_real)),
        # resample size = n_obs to form a predicted-density cloud at t_real.
        s_all = adata.obsm[cellstate_key][:, :n_dims].astype(np.float32)
        s_all_t = torch.from_numpy(s_all).to(device)
        t_norm = float(t_real) - t_min
        t_in = torch.full((s_all_t.shape[0], 1), t_norm * tsf, dtype=torch.float32, device=device)
        with torch.no_grad():
            logu = model.u(s_all_t, t_in).cpu().numpy().ravel()
        # Stable softmax over log_u
        logu = logu - logu.max()
        w = np.exp(logu)
        w = w / w.sum()
        n_resample = int(s_obs.shape[0])
        idx = rng.choice(s_all.shape[0], size=n_resample, replace=True, p=w)
        s_pred = s_all[idx]

        cap = args.w2_cap_cells
        if s_pred.shape[0] > cap:
            s_pred = s_pred[rng.choice(s_pred.shape[0], cap, replace=False)]
        if s_obs.shape[0] > cap:
            s_obs = s_obs[rng.choice(s_obs.shape[0], cap, replace=False)]
        w2 = float(fate_eval.compute_w2(s_pred, s_obs, device=device))
        out.append({"t_real": t_real, "n_obs": int(sub_obs.n_obs),
                    "n_pred": int(n_resample), "w2_holdout": w2})
    return out


# -----------------------------------------------------------------------------
# Stage GROWTH (tom_pos) — per (arm, seed, t_k) min/max g + interval slope
# -----------------------------------------------------------------------------
def stage_growth(row: dict, config, model, device, args) -> list[dict]:
    if row["dataset"] != "tom_pos":
        return []
    spec = DATASET_SPEC["tom_pos"]
    adata = get_adata("tom_pos")
    cellstate_key = config.dataset_config["cellstate_key"]
    n_dims = config.dataset_config["n_dimension"]
    tp_key = spec["tp_key"] if spec["tp_key"] in adata.obs.columns else _autoselect_tp_key(adata)
    tsf = float(model.time_scale_factor)
    t_obs_real = sorted(adata.obs[tp_key].unique().tolist())
    t_min = float(min(t_obs_real))

    counts = {t: int((adata.obs[tp_key] == t).sum()) for t in t_obs_real}
    log_n = {t: np.log(max(counts[t], 1)) for t in t_obs_real}
    out = []
    for k, t_real in enumerate(t_obs_real):
        sub = adata[adata.obs[tp_key] == t_real]
        s_np = sub.obsm[cellstate_key][:, :n_dims].astype(np.float32)
        s = torch.from_numpy(s_np).to(device)
        t_norm = float(t_real) - t_min
        t_in = torch.full((s.shape[0], 1), t_norm * tsf, dtype=torch.float32, device=device)
        with torch.no_grad():
            g = model.g(s, t_in).cpu().numpy().ravel()
        # interval slope to next observed tp (None at last tp)
        if k + 1 < len(t_obs_real):
            t_next = t_obs_real[k + 1]
            dt = float(t_next - t_real)
            g_int = float((log_n[t_next] - log_n[t_real]) / dt) if dt > 0 else float("nan")
        else:
            g_int = float("nan")
        out.append({
            "t_real": float(t_real),
            "n_cells": int(sub.n_obs),
            "g_min": float(np.min(g)),
            "g_max": float(np.max(g)),
            "g_p5": float(np.percentile(g, 5)),
            "g_p95": float(np.percentile(g, 95)),
            "g_median": float(np.median(g)),
            "g_int_to_next": g_int,
        })
    return out


# -----------------------------------------------------------------------------
# I/O — append-or-skip CSV writes for resumable runs
# -----------------------------------------------------------------------------
def append_rows(csv_path: Path, rows: list[dict], key_cols: list[str], overwrite: bool):
    """Append rows; skip ones whose key already exists unless overwrite=True."""
    if not rows:
        return
    if csv_path.exists() and not overwrite:
        existing = pd.read_csv(csv_path)
        existing_keys = {tuple(r[k] for k in key_cols) for _, r in existing.iterrows()}
        rows = [r for r in rows if tuple(r[k] for k in key_cols) not in existing_keys]
        if not rows:
            return
    df_new = pd.DataFrame(rows)
    if csv_path.exists() and not overwrite:
        df_new.to_csv(csv_path, mode="a", index=False, header=False)
    else:
        df_new.to_csv(csv_path, index=False)


def field_npz_path(dataset: str, arm: str, seed: int) -> Path:
    return FIELDS_DIR / f"{dataset}_{arm}_seed{seed}.npz"


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
def run(args):
    census = pd.read_csv(CENSUS)
    if args.datasets:
        census = census[census["dataset"].isin(args.datasets)]
    if args.arms:
        census = census[census["arm"].isin(args.arms)]
    if args.seeds:
        census = census[census["seed"].isin(args.seeds)]
    census = census.dropna(subset=["checkpoint_path", "run_config_path"])
    print(f"[start] {len(census)} (dataset, arm, seed) rows queued")
    print(f"[stages] {args.stages}")

    # Put sim_fn in the filename so different sim_fn runs don't collide.
    w2_csv = EVAL_DIR / f"w2_per_arm_{args.sim_fn}.csv"
    holdout_csv = EVAL_DIR / "tompos_holdout_w2.csv"
    growth_csv = EVAL_DIR / "tompos_growth_rate.csv"

    for i, (_, row) in enumerate(census.iterrows()):
        dataset = row["dataset"]
        arm = row["arm"]
        seed = int(row["seed"])
        ckpt_path = row["checkpoint_path"]
        cfg_path = row["run_config_path"]
        print(f"\n[{i+1}/{len(census)}] {dataset} / {arm} / seed_{seed}")
        if not os.path.isfile(ckpt_path):
            print(f"  [skip] missing ckpt: {ckpt_path}")
            continue
        try:
            config = load_config(cfg_path)
            model = load_model(ckpt_path, args.device)
        except Exception as e:
            print(f"  [error] load failed: {e}")
            continue

        if "w2" in args.stages:
            try:
                if dataset.startswith("klein"):
                    rows_w2 = stage_w2_klein(row.to_dict(), config, model, args.device, args)
                else:
                    rows_w2 = stage_w2_tompos(row.to_dict(), config, model, args.device, args)
                for r in rows_w2:
                    r.update({
                        "dataset": dataset, "arm": arm, "seed": seed,
                        "best_epoch": int(row["best_epoch"]) if pd.notna(row["best_epoch"]) else None,
                        "ckpt": os.path.basename(ckpt_path),
                    })
                append_rows(w2_csv, rows_w2,
                            ["dataset", "arm", "seed", "replicate"],
                            args.overwrite)
                if rows_w2:
                    mean_raw = np.mean([r["w2_raw"] for r in rows_w2])
                    print(f"  [w2]   raw_mean={mean_raw:.4f}  reps={len(rows_w2)}")
            except Exception as e:
                print(f"  [w2 error] {e}")

        if "fields" in args.stages:
            npz = field_npz_path(dataset, arm, seed)
            if npz.exists() and not args.overwrite:
                print(f"  [fields] skip (exists): {npz.name}")
            else:
                try:
                    arrays = stage_fields(row.to_dict(), config, model, args.device, args)
                    np.savez_compressed(npz, **arrays)
                    print(f"  [fields] -> {npz.name}  keys={len(arrays)}")
                except Exception as e:
                    print(f"  [fields error] {e}")

        if "holdout" in args.stages:
            try:
                rows_h = stage_holdout(row.to_dict(), config, model, args.device, args)
                for r in rows_h:
                    r.update({"dataset": dataset, "arm": arm, "seed": seed})
                append_rows(holdout_csv, rows_h, ["dataset", "arm", "seed", "t_real"], args.overwrite)
                if rows_h:
                    print(f"  [holdout] tps={[r['t_real'] for r in rows_h]} "
                          f"w2={[round(r['w2_holdout'],4) for r in rows_h]}")
            except Exception as e:
                print(f"  [holdout error] {e}")

        if "growth" in args.stages:
            try:
                rows_g = stage_growth(row.to_dict(), config, model, args.device, args)
                for r in rows_g:
                    r.update({"dataset": dataset, "arm": arm, "seed": seed})
                append_rows(growth_csv, rows_g, ["dataset", "arm", "seed", "t_real"], args.overwrite)
                if rows_g:
                    print(f"  [growth] tps={len(rows_g)}")
            except Exception as e:
                print(f"  [growth error] {e}")

        # Free GPU memory between rows
        del model
        torch.cuda.empty_cache()

    # -- final aggregation -----------------------------------------------------
    if "w2" in args.stages and w2_csv.exists():
        df = pd.read_csv(w2_csv)
        agg = df.groupby(["dataset", "arm", "seed"]).agg(
            w2_raw_mean=("w2_raw", "mean"),
            w2_scaled_mean=("w2_scaled", "mean"),
            n_reps=("replicate", "count"),
        ).reset_index()
        per_arm = agg.groupby(["dataset", "arm"]).agg(
            w2_raw=("w2_raw_mean", "mean"),
            w2_raw_std=("w2_raw_mean", "std"),
            w2_scaled=("w2_scaled_mean", "mean"),
            n_seeds=("seed", "count"),
        ).reset_index()
        baseline = per_arm[per_arm["arm"] == "baseline"][["dataset", "w2_raw"]].rename(
            columns={"w2_raw": "w2_raw_baseline"}
        )
        per_arm = per_arm.merge(baseline, on="dataset", how="left")
        per_arm["delta_w2_raw"] = per_arm["w2_raw"] - per_arm["w2_raw_baseline"]
        per_arm["abs_delta_w2"] = per_arm["delta_w2_raw"].abs()
        per_arm.to_csv(EVAL_DIR / f"w2_summary_{args.sim_fn}.csv", index=False)
        print(f"\n[summary] -> {EVAL_DIR / f'w2_summary_{args.sim_fn}.csv'}  ({len(per_arm)} arm rows)")

    print("\n[done]")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="*", default=None,
                   choices=["klein_OT", "klein_nonOT", "tom_pos"])
    p.add_argument("--arms", nargs="*", default=None,
                   help="filter to specific arm names (baseline, lambdaD_0, ...)")
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--stages", nargs="+", default=["w2", "fields", "holdout", "growth"],
                   choices=["w2", "fields", "holdout", "growth"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sim-fn", choices=["sb", "sde", "ode"], default="sb",
                   help="Simulator for the W2 stage (matches 03_eval_pdp_w2.py choices)")
    p.add_argument("--t-end-norm", type=float, default=4.0,
                   help="Integration end time in normalized units (klein day 6 = 4.0)")
    p.add_argument("--n-sims-replicates", type=int, default=3,
                   help="W2 replicates per arm (mean over replicates is the headline)")
    p.add_argument("--n-sims-per-cell", type=int, default=10,
                   help="SDE trajectories per starting cell within a replicate "
                        "(matches klein_w2_v1 / 03_eval_pdp_w2.py default)")
    p.add_argument("--n-steps", type=int, default=200, help="Euler-Maruyama steps")
    p.add_argument("--noise-scale", type=float, default=1.0)
    p.add_argument("--w2-cap-cells", type=int, default=5000,
                   help="Subsample to this many cells before W2 (entropic OT cost)")
    p.add_argument("--overwrite", action="store_true",
                   help="overwrite existing CSV rows / NPZ files")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
