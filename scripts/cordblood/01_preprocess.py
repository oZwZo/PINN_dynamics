#!/usr/bin/env python
"""01_preprocess.py — Cord blood (CLADES) AnnData preprocessing.

Analog of `scripts/pseudodynamics+/01_klein_preprocess.py`.

Inputs
------
- data/CordBlood/raw/CordBlood_Refine/adata_update.h5ad
- data/CordBlood/raw/CordBlood_Refine/CordBlood_Obs_Clones.pkl
- data/CordBlood/raw/CordBlood_Refine/data/kinetics_array_correction_factor.txt

Outputs
-------
- data/cordblood_addpop.h5ad  — canonical preprocessed file
- data/CordBlood/F_obs_day17.csv
- data/CordBlood/clone_proportions_day17.csv
- docs/cordblood/findings.md — appended per-fate train/val/test counts

Steps (see docs/cordblood/task_plan.md Phase C):
 1. Load.
 2. Cast `Timepoint` ('Day3'/'Day10'/'Day17') → int → `obs['timepoint_tx_days']`.
 3. PCA source: `X_pca` (50 dims) already present. No harmony in CLADES adata.
 4. Palantir DM + multiscaled → `obsm['DM_EigenVectors']`, `obsm['DM_EigenVectors_multiscaled']`.
 5. Clone barcode split (`np.random.seed(42)`):
    - Multi-cell barcoded clones → 80/10/10 by clone ID (train/val/test).
    - Singleton-barcoded and unbarcoded cells → `train_unbarcoded` (informative
      input but not used in clone-fate eval).
    - Enforce ≥5 clones per fate (at Day17) in both train and test.
 6. F_obs_day17.csv (one-hot fate by cell) + clone_proportions_day17.csv (per clone).
 7. delta_DM (palantir DM, 20 repeats avg) and delta_PC (X_pca, 20 repeats avg).
 8. z-score standardization fitted on TRAIN ONLY:
    - X_pca[:30] → obsm['X_pca_scaled'], uns['pca_scaler']
    - DM_EigenVectors → obsm['DM_EigenVectors_scaled'], uns['dm_scaler']
 9. uns['pop'] from kinetics_array_correction_factor.txt (per-timepoint totals,
    Poisson variance fallback unless replicate info is available).
10. Schema validation via `scripts/cordblood/_validate_adata.py`.
11. Save.
"""
from __future__ import annotations

import os
import sys
import pickle
import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import palantir

ROOT = Path("/rds/user/wz369/hpc-work/PINN_dynamics")
# Import the project's PINN package (aliases PINN.functions → PINN.tl, providing sample_deltax).
sys.path.insert(0, str(ROOT))
import PINN as pdp  # noqa: E402  (pdp.tl.sample_deltax lives in PINN/functions/reader_funs.py)

# Allow importing the schema validator from the same dir
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _validate_adata  # noqa: E402


RAW = ROOT / "data/CordBlood/raw/CordBlood_Refine"
ADATA_IN = RAW / "adata_update.h5ad"
CLONES_PKL = RAW / "CordBlood_Obs_Clones.pkl"
KINETICS_TXT = RAW / "data" / "kinetics_array_correction_factor.txt"

ADATA_OUT = ROOT / "data" / "cordblood_addpop.h5ad"
FOBS_OUT = ROOT / "data" / "CordBlood" / "F_obs_day17.csv"
CLONE_PROP_OUT = ROOT / "data" / "CordBlood" / "clone_proportions_day17.csv"
FINDINGS_OUT = ROOT / "docs" / "cordblood" / "findings.md"

# CLADES cell type list (from prepare_input.ipynb)
POPS = ['HSC/MPP 1', 'HSC/MPP 2', 'MEMP', 'Mast cell',
        'Early Erythroid', 'Mid Erythroid', 'Late Erythroid',
        'NMP', 'Mono precur', 'Monocyte', 'DC precursor', 'DC']
TIMES = [3, 10, 17]
SEED = 42
FATE_TP = 17                    # F_obs derived from Day17
N_PCA_DIMS = 30
MIN_CLONES_PER_FATE = 5


def log(msg):
    print(f"[01_preprocess] {msg}", flush=True)


def cast_timepoint(adata):
    """Day3/Day10/Day17 → int → obs['timepoint_tx_days']."""
    mapping = {"Day3": 3, "Day10": 10, "Day17": 17}
    tp_str = adata.obs["Timepoint"].astype(str)
    unknown = set(tp_str.unique()) - set(mapping.keys())
    if unknown:
        raise ValueError(f"Unknown Timepoint values: {unknown}")
    adata.obs["timepoint_tx_days"] = tp_str.map(mapping).astype(int)
    log(f"  timepoint_tx_days unique = {sorted(adata.obs['timepoint_tx_days'].unique())}")


def run_palantir_dm(adata, pca_key):
    """Run palantir diffusion maps + multiscale.

    Palantir's `run_diffusion_maps` writes `obsm['DM_EigenVectors']` whose first
    column is the trivial steady-state (constant) eigenvector. We drop it so
    every retained dim has non-zero variance — otherwise downstream
    standardization explodes.
    """
    log(f"  palantir.run_diffusion_maps(pca_key={pca_key!r}) ...")
    palantir.utils.run_diffusion_maps(adata, pca_key=pca_key)
    palantir.utils.determine_multiscale_space(adata)
    if "DM_EigenVectors" not in adata.obsm:
        raise RuntimeError(
            f"palantir did not produce DM_EigenVectors. obsm: {list(adata.obsm.keys())}"
        )
    full = adata.obsm["DM_EigenVectors"]
    full_std = full.std(axis=0)
    log(f"  raw DM_EigenVectors shape = {full.shape}, per-dim std = {full_std.round(6).tolist()}")

    # Drop any near-constant leading column(s) (palantir's steady-state eigenvector).
    drop_mask = full_std < 1e-6
    if drop_mask.any():
        keep = ~drop_mask
        n_dropped = int(drop_mask.sum())
        log(f"  dropping {n_dropped} zero-variance DM eigenvector(s): "
            f"indices {np.where(drop_mask)[0].tolist()}")
        adata.obsm["DM_EigenVectors"] = full[:, keep]

    # Same defensive trim on multiscaled (palantir already filters by eigenvalue,
    # but be safe — and we don't ship the trivial column to downstream methods).
    if "DM_EigenVectors_multiscaled" in adata.obsm:
        ms = adata.obsm["DM_EigenVectors_multiscaled"]
        ms_std = ms.std(axis=0)
        ms_drop = ms_std < 1e-6
        if ms_drop.any():
            log(f"  dropping {int(ms_drop.sum())} zero-variance multiscaled dim(s)")
            adata.obsm["DM_EigenVectors_multiscaled"] = ms[:, ~ms_drop]

    n_dim = adata.obsm["DM_EigenVectors"].shape[1]
    log(f"  DM_EigenVectors shape (after trim) = {adata.obsm['DM_EigenVectors'].shape}, "
        f"multiscaled = {adata.obsm.get('DM_EigenVectors_multiscaled', np.array([])).shape}")
    return n_dim


def _select_test_clones_strategy_I_prime(adata, *, seed=SEED, target_max_delta=5.0):
    """Strategy I' — stratified test split, restricted to multi-timepoint clones,
    with greedy drop of skew-contributing outliers.

    Selection pipeline (deterministic given seed):
      1. Restrict candidates to multi-cell + multi-timepoint clones.
      2. Compute each clone's primary Day-17 fate (argmax over its Day-17 cells).
      3. For each primary fate, randomly sample 10 % of clones → initial test set.
      4. Greedy drop the clone whose removal most reduces
         max_fate |train_pct − test_pct| (Day-17 cell-level proportions),
         until improvement stalls OR max |delta| < target_max_delta.

    Returns the final test-clone set.
    """
    clones_series = adata.obs["clones"].astype(str)
    is_barcoded = clones_series != "Clone_"
    clone_sizes = clones_series[is_barcoded].value_counts()
    multi_cell = set(clone_sizes[clone_sizes > 1].index)

    n_tp_per_clone = (
        adata.obs[is_barcoded]
        .groupby(clones_series[is_barcoded])["timepoint_tx_days"]
        .nunique()
    )
    multi_tp = set(n_tp_per_clone[n_tp_per_clone >= 2].index)
    eligible = multi_tp & multi_cell

    d17_mask = adata.obs["timepoint_tx_days"] == FATE_TP
    sub = adata.obs[d17_mask & is_barcoded].copy()
    sub["clone"] = sub["clones"].astype(str)
    sub_eligible = sub[sub["clone"].isin(eligible)]
    primary_fate = (
        sub_eligible.groupby("clone")["def_lab"]
        .agg(lambda x: x.value_counts().idxmax())
    )

    rng = np.random.RandomState(seed)
    initial_test = set()
    for fate, group_clones in primary_fate.groupby(primary_fate):
        n = max(1, int(round(0.10 * len(group_clones))))
        initial_test.update(
            rng.choice(group_clones.index.tolist(), size=n, replace=False).tolist()
        )
    log(f"  Strategy I initial test: {len(initial_test):,} clones (stratified per primary D17 fate)")

    # ── Greedy drop to balance Day-17 fate proportions ──
    fate_categories = (
        list(adata.obs["def_lab"].cat.categories)
        if hasattr(adata.obs["def_lab"], "cat")
        else sorted(adata.obs["def_lab"].unique())
    )

    def fate_pcts(clones_set):
        m = (clones_series.isin(clones_set)) & d17_mask
        counts = adata.obs.loc[m, "def_lab"].value_counts()
        counts = counts.reindex(fate_categories, fill_value=0)
        total = counts.sum()
        if total == 0:
            return counts.astype(float) * 0
        return (counts / total * 100).astype(float)

    def max_abs_delta(test_set):
        return (fate_pcts(test_set) - fate_pcts(multi_cell - test_set)).abs().max()

    cur_test = set(initial_test)
    cur_delta = max_abs_delta(cur_test)
    log(f"  Strategy I baseline max |delta| = {cur_delta:.2f}")

    dropped = []
    for step in range(50):
        if cur_delta < target_max_delta:
            log(f"  greedy stop: max |delta| {cur_delta:.2f} < target {target_max_delta}")
            break
        # Try removing each clone in cur_test, find the one that gives lowest max|delta|
        best = (None, cur_delta)
        for c in cur_test:
            new_delta = max_abs_delta(cur_test - {c})
            if new_delta < best[1]:
                best = (c, new_delta)
        if best[0] is None:
            log(f"  greedy stop: no further improvement at step {step} (max |delta|={cur_delta:.2f})")
            break
        cur_test.discard(best[0])
        dropped.append((best[0], primary_fate[best[0]], best[1]))
        cur_delta = best[1]
        log(f"  step {step}: dropped {best[0]} (primary={primary_fate[best[0]]}) "
            f"→ max |delta|={cur_delta:.2f}, {len(cur_test):,} test clones left")

    log(f"  Strategy I' FINAL: {len(cur_test):,} test clones, max |delta|={cur_delta:.2f}, "
        f"dropped {len(dropped)}")
    return cur_test, dropped


def split_by_clone(adata, *, seed=SEED):
    """Build obs['split'] using Strategy I':
      - test = greedy-balanced stratified sample of multi-timepoint clones (37-ish)
      - val  = ~10 % of remaining multi-cell clones (random, same seed)
      - train = remaining multi-cell clones
      - train_unbarcoded = singletons + unbarcoded sentinel cells
    """
    clones_series = adata.obs["clones"].astype(str)
    is_barcoded = clones_series != "Clone_"
    clone_sizes = clones_series[is_barcoded].value_counts()
    multi_cell = set(clone_sizes[clone_sizes > 1].index)
    log(f"  multi-cell barcoded clones: {len(multi_cell):,}  "
        f"(singletons: {(clone_sizes == 1).sum():,})")

    # Strategy I' test selection (deterministic via seed)
    test_clones, dropped_outliers = _select_test_clones_strategy_I_prime(adata, seed=seed)

    # Val: 10 % of remaining multi-cell clones (random, separate RNG state)
    remaining = sorted(multi_cell - test_clones)
    rng_val = np.random.RandomState(seed + 1)
    rng_val.shuffle(remaining)
    n_val = int(round(0.10 * len(remaining)))
    val_clones = set(remaining[:n_val])
    train_clones = set(remaining[n_val:])
    log(f"  clone counts → train: {len(train_clones):,}  val: {len(val_clones):,}  test: {len(test_clones):,}")

    split = np.empty(adata.n_obs, dtype=object)
    for i, c in enumerate(clones_series.values):
        if c in train_clones:
            split[i] = "train"
        elif c in val_clones:
            split[i] = "val"
        elif c in test_clones:
            split[i] = "test"
        else:
            split[i] = "train_unbarcoded"
    adata.obs["split"] = pd.Categorical(split, categories=["train", "val", "test", "train_unbarcoded"])
    log("  split counts:")
    for s, n in adata.obs["split"].value_counts().items():
        log(f"    {s:18s} {n:>7,}")

    # Compat column for legacy Klein eval scripts that hardcode `obs.Well == 2`:
    # OT-CFM / SF2M / TIGON / PRESCIENT 03_evaluate.py all use `obs.Well == 2`
    # to mask the test set without a --well_col flag. Mirror that contract.
    adata.obs["Well"] = np.where(np.asarray(split) == "test", 2, 0).astype(int)
    log(f"  obs['Well'] compat column added: "
        f"{{0: train+val+unbarcoded={(adata.obs['Well'] == 0).sum():,}, "
        f"2: test={(adata.obs['Well'] == 2).sum():,}}}")

    return train_clones, val_clones, test_clones


def fate_balance_check(adata, train_clones, test_clones, *, min_per_fate=MIN_CLONES_PER_FATE):
    """For each fate at Day17, count how many train and test clones contribute."""
    day_mask = adata.obs["timepoint_tx_days"] == FATE_TP
    sub = adata.obs[day_mask].copy()
    sub["clone"] = sub["clones"].astype(str)
    counts = {}
    for fate in sub["def_lab"].cat.categories if hasattr(sub["def_lab"], "cat") else sub["def_lab"].unique():
        rows = sub[sub["def_lab"] == fate]
        train_n = rows[rows["clone"].isin(train_clones)]["clone"].nunique()
        test_n = rows[rows["clone"].isin(test_clones)]["clone"].nunique()
        counts[fate] = (train_n, test_n)
    log(f"  fate balance at Day{FATE_TP} (train_clones, test_clones):")
    deficient = []
    for fate, (a, b) in counts.items():
        marker = "" if (a >= min_per_fate and b >= min_per_fate) else "  ← <{}".format(min_per_fate)
        log(f"    {fate:22s} train={a:>4d}  test={b:>4d}{marker}")
        if a < min_per_fate or b < min_per_fate:
            deficient.append((fate, a, b))
    return counts, deficient


def build_F_obs(adata, train_clones, val_clones, test_clones):
    """F_obs_day17.csv (one-hot fate per Day17 cell) + clone_proportions_day17.csv."""
    day_mask = (adata.obs["timepoint_tx_days"] == FATE_TP).values
    is_barcoded = (adata.obs["clones"].astype(str) != "Clone_").values
    eval_mask = day_mask & is_barcoded
    sub = adata.obs.loc[eval_mask].copy()

    # one-hot fate per cell
    fate_cats = list(adata.obs["def_lab"].cat.categories) if hasattr(adata.obs["def_lab"], "cat") else sorted(adata.obs["def_lab"].unique())
    F_obs = pd.get_dummies(sub["def_lab"]).reindex(columns=fate_cats, fill_value=0).astype(int)
    F_obs.index = sub.index
    F_obs.to_csv(FOBS_OUT)
    log(f"  wrote {FOBS_OUT}  shape={F_obs.shape}")

    # per-clone proportions at Day17 (multi-cell barcoded clones only)
    multi_cell = train_clones | val_clones | test_clones
    sub_m = sub[sub["clones"].astype(str).isin(multi_cell)]
    clone_prop = (
        sub_m.assign(clone=sub_m["clones"].astype(str))
             .groupby("clone")["def_lab"]
             .value_counts(normalize=True)
             .unstack(fill_value=0.0)
             .reindex(columns=fate_cats, fill_value=0.0)
    )
    clone_prop.to_csv(CLONE_PROP_OUT)
    log(f"  wrote {CLONE_PROP_OUT}  shape={clone_prop.shape}")
    return F_obs, clone_prop


def _sample_deltax_stack(adata, xkey, pseudotime_key, n_calls):
    """Call sample_deltax `n_calls` times and return stacked deltas.

    PINN.tl.sample_deltax returns a list of n_cells arrays each of shape
    (n_repeat, n_dims) where n_repeat=10 by library default. We average over
    both the outer call axis AND the inner n_repeat axis to get one deltax
    per cell.

    Returns: ndarray shape (n_cells, n_dims) — already averaged.
    """
    accum = None  # running sum over both call and n_repeat axes
    count = 0
    for k in range(n_calls):
        dlt, _ = pdp.tl.sample_deltax(
            adata, xkey=xkey, pseudotimekey=pseudotime_key, progressbar=False
        )
        # dlt is a list of length n_cells; each entry shape (n_repeat, n_dims).
        # Stack to a numpy array of shape (n_cells, n_repeat, n_dims).
        arr = np.stack(dlt)
        if accum is None:
            accum = np.zeros((arr.shape[0], arr.shape[2]), dtype=np.float64)
        # Mean over n_repeat axis for this call → (n_cells, n_dims)
        accum += arr.mean(axis=1)
        count += 1
        if (k + 1) % 5 == 0:
            log(f"    sample_deltax call {k + 1}/{n_calls} done")
    return (accum / count).astype(np.float32)


def compute_deltas(adata, dm_key, pca_key, *, repeats=20, pseudotime_key="dpt_pseudotime"):
    """delta_DM (palantir DM) and delta_PC (X_pca) via pdp.tl.sample_deltax.

    `pseudotime_key` defaults to 'dpt_pseudotime' for CLADES (the CLADES adata
    has DPT but NOT palantir_pseudotime, which is sample_deltax's library default).
    Local PINN.tl.sample_deltax uses n_repeat=10 internally, so we get
    20×10 = 200 effective samples per cell at default.
    """
    log(f"  delta_DM: {repeats}x sample_deltax(xkey={dm_key!r}, "
        f"pseudotimekey={pseudotime_key!r}) ...")
    adata.obsm["delta_DM"] = _sample_deltax_stack(adata, dm_key, pseudotime_key, repeats)
    log(f"  delta_DM shape = {adata.obsm['delta_DM'].shape}")

    log(f"  delta_PC: {repeats}x sample_deltax(xkey={pca_key!r}, "
        f"pseudotimekey={pseudotime_key!r}) ...")
    delta_pc_full = _sample_deltax_stack(adata, pca_key, pseudotime_key, repeats)
    # Only keep the first N_PCA_DIMS (matches X_pca_scaled width)
    adata.obsm["delta_PC"] = delta_pc_full[:, :N_PCA_DIMS]
    log(f"  delta_PC shape = {adata.obsm['delta_PC'].shape}")


def standardize_train_only(adata, source_key, scaler_key, scaled_key, n_dims):
    """Fit mean/std on TRAIN split only, apply to all cells. Fail loud on zero variance."""
    train_mask = (adata.obs["split"] == "train").values
    if train_mask.sum() == 0:
        raise ValueError("No 'train' cells — cannot fit scaler.")
    raw = adata.obsm[source_key]
    if raw.shape[1] < n_dims:
        raise ValueError(f"obsm[{source_key!r}] has {raw.shape[1]} dims, need {n_dims}")
    raw = raw[:, :n_dims].astype(np.float64)
    raw_train = raw[train_mask]
    mean = raw_train.mean(axis=0)
    std = raw_train.std(axis=0)
    if (std < 1e-6).any():
        bad = np.where(std < 1e-6)[0].tolist()
        raise ValueError(
            f"Zero-variance dims in TRAIN cells for {source_key!r}: {bad}. "
            f"Refusing to clip silently."
        )
    adata.uns[scaler_key] = {
        "mean": mean.astype(np.float64),
        "std": std.astype(np.float64),
        "n_dims": int(n_dims),
        "source_key": str(source_key),
        "fit_on": "train",
        "n_train_cells": int(train_mask.sum()),
    }
    adata.obsm[scaled_key] = ((raw - mean) / std).astype(np.float32)
    log(f"  uns[{scaler_key!r}]  fit on {train_mask.sum():,} train cells, "
        f"applied to all {adata.n_obs:,} cells → obsm[{scaled_key!r}] {adata.obsm[scaled_key].shape}")


def build_pop_uns(adata):
    """uns['pop'] from kinetics_array_correction_factor.txt.

    The pseudodynamics+ reader iterates `uns['pop']` and indexes every value by
    timepoint_idx — so EVERY value here MUST be an indexable array (length = n_tp).
    The variance-source annotation is stashed in a sibling `uns['pop_meta']` key
    so it doesn't break the reader.

    The last row of the (13, 36) array — when reshaped to (13, 3, 12) — is the
    "all-clones" row: per-timepoint per-population totals after FACS correction.
    """
    kinetics_flat = np.loadtxt(KINETICS_TXT)
    if kinetics_flat.shape != (13, 36):
        raise ValueError(f"Unexpected kinetics shape: {kinetics_flat.shape}, expected (13, 36)")
    kinetics = kinetics_flat.reshape(13, 3, 12)
    pop_per_tp_pop = kinetics[-1]  # (3, 12): per-tp per-population totals
    per_tp_total = pop_per_tp_pop.sum(axis=1)  # (3,)
    log(f"  uns['pop'] per-tp totals: {dict(zip(TIMES, per_tp_total.tolist()))}")
    adata.uns["pop"] = {
        "t":     np.asarray(TIMES, dtype=np.float64),
        "mean":  per_tp_total.astype(np.float64),
        "std":   np.sqrt(per_tp_total).astype(np.float64),  # Poisson fallback — no replicate variance available
        "var":   per_tp_total.astype(np.float64),
        "n_lib": per_tp_total.astype(np.float64),
    }
    # Annotations that are NOT per-timepoint arrays live in a sibling key.
    adata.uns["pop_meta"] = {
        "variance_source": "poisson_fallback",
        "source_file":     "kinetics_array_correction_factor.txt",
    }


def append_findings(adata, train_clones, val_clones, test_clones, fate_counts, deficient):
    n_total = adata.n_obs
    split_counts = adata.obs["split"].value_counts().to_dict()

    lines = [
        "",
        f"## Phase C preprocessing summary (auto-generated {pd.Timestamp.now().isoformat()})",
        "",
        f"- Source adata: `{ADATA_IN}`",
        f"- Output adata: `{ADATA_OUT}`",
        f"- Total cells preserved: {n_total:,}",
        f"- Genes: {adata.n_vars:,}",
        "",
        "### Split summary",
        "",
        "| split | cells |",
        "|---|---|",
    ]
    for k in ("train", "val", "test", "train_unbarcoded"):
        lines.append(f"| {k} | {split_counts.get(k, 0):,} |")
    lines += [
        "",
        f"- Multi-cell barcoded clones: train={len(train_clones):,}  "
        f"val={len(val_clones):,}  test={len(test_clones):,}",
        "",
        "### Per-fate clone counts at Day17 (multi-cell clones)",
        "",
        "| fate | train_clones | test_clones |",
        "|---|---|---|",
    ]
    for fate, (a, b) in fate_counts.items():
        lines.append(f"| {fate} | {a} | {b} |")
    if deficient:
        lines.append("")
        lines.append(f"**WARN**: {len(deficient)} fates have <{MIN_CLONES_PER_FATE} clones in train or test:")
        for f, a, b in deficient:
            lines.append(f"- {f}: train={a}, test={b}")
    lines += [
        "",
        f"### PCA / DM",
        f"- pca_source = `{adata.uns['pca_source']}`",
        f"- dm_n_dims = {adata.uns['dm_n_dims']}",
        f"- uns['pca_scaler']: mean={adata.uns['pca_scaler']['mean'].shape}, "
        f"  std={adata.uns['pca_scaler']['std'].shape}, fit_on=train, "
        f"  n_train_cells={adata.uns['pca_scaler']['n_train_cells']:,}",
        f"- uns['dm_scaler']: mean={adata.uns['dm_scaler']['mean'].shape}, "
        f"  std={adata.uns['dm_scaler']['std'].shape}",
        "",
        f"### Population (uns['pop'])",
        f"- variance_source: `{adata.uns['pop']['variance_source']}` "
        f"(Poisson std=sqrt(mean); no replicate variance in CLADES kinetics file)",
        f"- t = {adata.uns['pop']['t'].tolist()}",
        f"- mean = {adata.uns['pop']['mean'].tolist()}",
        "",
    ]
    with open(FINDINGS_OUT, "a") as fh:
        fh.write("\n".join(lines))
    log(f"  appended summary to {FINDINGS_OUT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Run all steps except writing outputs.")
    args = parser.parse_args()

    log("Loading adata ...")
    adata = sc.read_h5ad(ADATA_IN)
    log(f"  n_obs={adata.n_obs}  n_vars={adata.n_vars}")

    # Step 2 — timepoint
    log("Step 2 — timepoint_tx_days")
    cast_timepoint(adata)

    # Step 3 — PCA source (already present as X_pca[50])
    log("Step 3 — PCA source check")
    if "X_pca_harmony" in adata.obsm:
        pca_source = "X_pca_harmony"
    elif "X_pca" in adata.obsm:
        pca_source = "X_pca"
    else:
        raise RuntimeError("Neither X_pca nor X_pca_harmony in obsm")
    if adata.obsm[pca_source].shape[1] < N_PCA_DIMS:
        raise RuntimeError(f"{pca_source} has {adata.obsm[pca_source].shape[1]} dims, "
                           f"need at least {N_PCA_DIMS}")
    adata.uns["pca_source"] = pca_source
    log(f"  pca_source = {pca_source}  ({adata.obsm[pca_source].shape})")

    # Step 4 — palantir DM
    log("Step 4 — palantir DM")
    dm_n = run_palantir_dm(adata, pca_key=pca_source)
    adata.uns["dm_n_dims"] = int(dm_n)

    # Step 5 — split by clone (before scaler fit!)
    log("Step 5 — split by clone barcode (seed=42)")
    train_clones, val_clones, test_clones = split_by_clone(adata, seed=SEED)
    fate_counts, deficient = fate_balance_check(adata, train_clones, test_clones)
    if deficient:
        log(f"  WARN: {len(deficient)} fates fall below "
            f"MIN_CLONES_PER_FATE={MIN_CLONES_PER_FATE}. Proceeding anyway "
            "— rare fates documented in findings.md.")

    # Step 6 — F_obs + clone_proportions
    log(f"Step 6 — F_obs / clone_proportions at Day{FATE_TP}")
    if not args.dry_run:
        FOBS_OUT.parent.mkdir(parents=True, exist_ok=True)
        build_F_obs(adata, train_clones, val_clones, test_clones)
    else:
        log("  --dry-run: skipping F_obs writes")

    # Step 7 — delta vectors (do this BEFORE standardization so deltas are in raw embedding space)
    log("Step 7 — delta vectors")
    compute_deltas(adata, dm_key="DM_EigenVectors", pca_key=pca_source, repeats=20)

    # Step 8 — z-score standardization, fit on train only
    log("Step 8 — z-score standardization (TRAIN ONLY)")
    standardize_train_only(
        adata,
        source_key=pca_source,
        scaler_key="pca_scaler",
        scaled_key=f"{pca_source}_scaled",
        n_dims=N_PCA_DIMS,
    )
    # Also alias to "X_pca_scaled" if pca_source was harmony — methods read this name.
    # (For CLADES, pca_source = X_pca, so source_key + '_scaled' == 'X_pca_scaled' already.)
    if pca_source != "X_pca" and "X_pca_scaled" not in adata.obsm:
        adata.obsm["X_pca_scaled"] = adata.obsm[f"{pca_source}_scaled"]

    standardize_train_only(
        adata,
        source_key="DM_EigenVectors",
        scaler_key="dm_scaler",
        scaled_key="DM_EigenVectors_scaled",
        n_dims=dm_n,
    )

    # Step 9 — uns['pop']
    log("Step 9 — uns['pop'] from kinetics file")
    build_pop_uns(adata)

    # Step 10 — schema check
    log("Step 10 — schema validation")
    _validate_adata.validate(adata)
    log("  schema OK")

    # Step 11 — write
    if args.dry_run:
        log("--dry-run: skipping h5ad write")
    else:
        log(f"Step 11 — writing {ADATA_OUT}")
        ADATA_OUT.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(ADATA_OUT)
        log(f"  wrote {ADATA_OUT}  ({ADATA_OUT.stat().st_size / 1e6:.1f} MB)")

    # Append findings
    if not args.dry_run:
        append_findings(adata, train_clones, val_clones, test_clones, fate_counts, deficient)

    log("DONE")


if __name__ == "__main__":
    main()
