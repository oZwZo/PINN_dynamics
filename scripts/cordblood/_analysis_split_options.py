#!/usr/bin/env python
"""Compare candidate train/test split strategies at the Day-17 CELL level.

Outputs a short report:
  - Current split (random 80/10/10 by clone): cell counts per def_lab at Day 17.
  - Alternative A: hold out only "multipotent" clones (clones that span >=2
    Day-17 fates AND have >= N cells total).
  - Alternative B: hold out only "large" clones (>= N cells, regardless of fate
    diversity).
  - Alternative C: combo (multipotent OR large).

Read-only — does not modify the h5ad.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc

PATH = "/rds/user/wz369/hpc-work/PINN_dynamics/data/cordblood_addpop.h5ad"
LARGE_THR = 20   # >= 20 cells = "large"
MIN_FATES = 2    # >= 2 unique Day-17 fates = "multipotent"


def day17_celltype_counts(adata, clones_in_set):
    """Cell-level Day-17 fate counts for the given set of clones."""
    mask = (
        (adata.obs["timepoint_tx_days"] == 17)
        & adata.obs["clones"].astype(str).isin(clones_in_set)
    )
    return adata.obs.loc[mask, "def_lab"].value_counts()


def temporal_coverage(adata, clones_in_set):
    """For each timepoint, count how many test clones have ≥1 cell there."""
    out = {}
    for tp in [3, 10, 17]:
        mask = (
            (adata.obs["timepoint_tx_days"] == tp)
            & adata.obs["clones"].astype(str).isin(clones_in_set)
        )
        clones_at_tp = adata.obs.loc[mask, "clones"].astype(str).nunique()
        cells_at_tp = mask.sum()
        out[tp] = (int(clones_at_tp), int(cells_at_tp))
    # Multi-timepoint coverage
    counts_by_clone = (
        adata.obs[adata.obs["clones"].astype(str).isin(clones_in_set)]
        .groupby(adata.obs["clones"].astype(str))["timepoint_tx_days"]
        .nunique()
    )
    return out, counts_by_clone


def describe(adata, train_clones, test_clones, label):
    print(f"\n=== {label} ===")
    print(f"  train clones={len(train_clones):,}  test clones={len(test_clones):,}")
    # Temporal coverage of test clones
    tcov, tcounts = temporal_coverage(adata, test_clones)
    print(f"  TEST clone temporal coverage:")
    for tp, (nc, ncells) in tcov.items():
        print(f"    Day{tp:>2}: {nc:>3d} unique clones, {ncells:>5d} cells")
    print(f"    test clones at >=2 timepoints: {int((tcounts >= 2).sum())}/{len(test_clones)}")
    print(f"    test clones at all 3 timepoints: {int((tcounts == 3).sum())}/{len(test_clones)}")
    print(f"    test clones at Day17 ONLY: {int((tcounts == 1).sum() - ((tcounts == 1) & (tcounts.index.isin(set()))).sum())}")
    train_d17 = day17_celltype_counts(adata, train_clones)
    test_d17 = day17_celltype_counts(adata, test_clones)
    df = pd.DataFrame({
        "train_d17_cells": train_d17,
        "test_d17_cells": test_d17,
    }).fillna(0).astype(int)
    df["train_pct"] = (df["train_d17_cells"] / max(df["train_d17_cells"].sum(), 1) * 100).round(1)
    df["test_pct"] = (df["test_d17_cells"] / max(df["test_d17_cells"].sum(), 1) * 100).round(1)
    df["pct_delta"] = (df["test_pct"] - df["train_pct"]).round(1)
    df = df.sort_values("train_d17_cells", ascending=False)
    print(df.to_string())
    print(f"  total Day-17 train cells (these clones): {df['train_d17_cells'].sum():,}")
    print(f"  total Day-17 test cells (these clones):  {df['test_d17_cells'].sum():,}")


def main():
    print(f"Loading {PATH} ...")
    adata = sc.read_h5ad(PATH)
    clones = adata.obs["clones"].astype(str)
    is_barcoded = clones != "Clone_"

    # Clone size (total cells across all timepoints)
    clone_size = clones[is_barcoded].value_counts()
    multi_cell = clone_size[clone_size > 1].index

    # Per-clone Day-17 fate count (number of distinct def_lab values at Day 17)
    d17_mask = adata.obs["timepoint_tx_days"] == 17
    d17_clones = clones[d17_mask & is_barcoded]
    d17_fates = adata.obs.loc[d17_mask & is_barcoded, "def_lab"].astype(str)
    fates_per_clone = (
        pd.DataFrame({"clone": d17_clones.values, "fate": d17_fates.values})
        .groupby("clone")["fate"].nunique()
    )

    # Day-17 cells per clone
    d17_cells_per_clone = d17_clones.value_counts()

    print(f"\nSummary:")
    print(f"  multi-cell barcoded clones (any time): {len(multi_cell):,}")
    print(f"  multi-cell clones present at Day 17:   {len(d17_cells_per_clone):,}")
    print(f"  clones with >=1 Day-17 cell:           {len(d17_cells_per_clone):,}")
    print(f"  clones with >={MIN_FATES} distinct Day-17 fates (multipotent): "
          f"{int((fates_per_clone >= MIN_FATES).sum()):,}")
    print(f"  clones with >={LARGE_THR} total cells (large):                 "
          f"{int((clone_size >= LARGE_THR).sum()):,}")
    print(f"  clones with >={LARGE_THR} Day-17 cells (large at d17):         "
          f"{int((d17_cells_per_clone >= LARGE_THR).sum()):,}")
    print(f"  clones with >={LARGE_THR} cells AND >={MIN_FATES} fates (large+multipotent): "
          f"{int(((clone_size >= LARGE_THR) & (fates_per_clone.reindex(clone_size.index).fillna(0) >= MIN_FATES)).sum()):,}")

    # ── Current split (whatever's in adata.obs['split']) ──
    current_train = set(clones[adata.obs["split"] == "train"].unique())
    current_test = set(clones[adata.obs["split"] == "test"].unique())
    current_train -= {"Clone_"}
    current_test -= {"Clone_"}
    describe(adata, current_train, current_test,
             "Current split (random 80/10/10 by multi-cell clone)")

    # ── Alternative A: hold out ONLY multipotent clones (≥2 Day-17 fates) ──
    multipotent = set(fates_per_clone[fates_per_clone >= MIN_FATES].index)
    train_A = set(multi_cell) - multipotent
    test_A = multipotent
    describe(adata, train_A, test_A,
             f"A: hold out multipotent clones (>={MIN_FATES} Day-17 fates)")

    # ── Alternative B: hold out ONLY large clones (>=N cells total) ──
    large = set(clone_size[clone_size >= LARGE_THR].index)
    train_B = set(multi_cell) - large
    test_B = large
    describe(adata, train_B, test_B,
             f"B: hold out large clones (>={LARGE_THR} total cells)")

    # ── Alternative C: hold out clones that are EITHER multipotent OR large ──
    test_C = multipotent | large
    train_C = set(multi_cell) - test_C
    describe(adata, train_C, test_C,
             f"C: hold out multipotent OR large clones")

    # ── Alternative D: hold out only clones that are BOTH multipotent AND large ──
    test_D = multipotent & large
    train_D = set(multi_cell) - test_D
    describe(adata, train_D, test_D,
             f"D: hold out clones that are multipotent AND large")

    # ── Alternative E: random 10% of multipotent clones ──
    rng = np.random.RandomState(42)
    mp_list = sorted(multipotent)
    n_test_mp = max(1, int(round(0.10 * len(mp_list))))
    test_E = set(rng.choice(mp_list, size=n_test_mp, replace=False).tolist())
    train_E = set(multi_cell) - test_E
    describe(adata, train_E, test_E,
             f"E: random 10% of multipotent clones held out ({n_test_mp} clones)")

    # ── Alternative F: stratified by def_lab at Day17 — 10% of clones per fate ──
    # For each fate, take the clones whose ARGMAX(Day17 fate count) == that fate,
    # then sample 10% of them. Provides per-fate test coverage by construction.
    sub = adata.obs[d17_mask & is_barcoded].copy()
    sub["clone"] = sub["clones"].astype(str)
    primary_fate = (
        sub.groupby("clone")["def_lab"]
           .agg(lambda x: x.value_counts().idxmax())
    )
    test_F = set()
    for fate, group_clones in primary_fate.groupby(primary_fate):
        n = max(1, int(round(0.10 * len(group_clones))))
        test_F.update(rng.choice(group_clones.index.tolist(), size=n, replace=False).tolist())
    train_F = set(multi_cell) - test_F
    describe(adata, train_F, test_F,
             f"F: stratified 10% per primary-Day17-fate ({len(test_F)} clones)")

    # ── Alternative G: stratified F + boost multipotent representation ──
    # Take 10% of clones per primary fate + add 10% of multipotent clones (set union).
    rng2 = np.random.RandomState(7)
    mp_sample = set(rng2.choice(sorted(multipotent), size=max(1, int(0.10 * len(multipotent))), replace=False).tolist())
    test_G = test_F | mp_sample
    train_G = set(multi_cell) - test_G
    describe(adata, train_G, test_G,
             f"G: stratified F + 10% multipotent ({len(test_G)} clones)")

    # ── Counts of multi-timepoint clones in the pool ──
    n_tp_per_clone = (
        adata.obs[is_barcoded]
        .groupby(clones[is_barcoded])["timepoint_tx_days"]
        .nunique()
    )
    multi_tp = set(n_tp_per_clone[n_tp_per_clone >= 2].index)
    all3_tp = set(n_tp_per_clone[n_tp_per_clone == 3].index)
    multi_tp_multi_cell = multi_tp & set(multi_cell)
    print(f"\n[summary of the clone pool]")
    print(f"  multi-cell clones: {len(multi_cell):,}")
    print(f"  multi-cell + multi-timepoint clones (≥2 tps): {len(multi_tp_multi_cell):,}")
    print(f"  multi-cell + all-3-timepoint clones: {len(all3_tp & set(multi_cell)):,}")
    print(f"  multi-cell + multipotent (≥2 D17 fates) clones: {len(multipotent):,}")
    print(f"  multi-cell + multipotent + multi-timepoint: {len(multipotent & multi_tp):,}")

    # ── Alternative H: hold out 10% of (multipotent AND multi-timepoint) clones ──
    mp_mt = multipotent & multi_tp
    n_test_h = max(1, int(round(0.10 * len(mp_mt))))
    rng3 = np.random.RandomState(42)
    test_H = set(rng3.choice(sorted(mp_mt), size=n_test_h, replace=False).tolist())
    train_H = set(multi_cell) - test_H
    describe(adata, train_H, test_H,
             f"H: 10% of (multipotent AND multi-timepoint) clones ({n_test_h} clones)")

    # ── Alternative I: stratified F restricted to multi-timepoint clones ──
    # Only pick test clones that are observed at >=2 timepoints (so per-clone
    # forward-simulation eval works).
    eligible_clones = multi_tp & set(multi_cell)
    sub_eligible = sub[sub["clone"].isin(eligible_clones)]
    primary_eligible = (
        sub_eligible.groupby("clone")["def_lab"]
            .agg(lambda x: x.value_counts().idxmax())
    )
    rng4 = np.random.RandomState(42)
    test_I = set()
    for fate, group_clones in primary_eligible.groupby(primary_eligible):
        n = max(1, int(round(0.10 * len(group_clones))))
        test_I.update(rng4.choice(group_clones.index.tolist(), size=n, replace=False).tolist())
    train_I = set(multi_cell) - test_I
    describe(adata, train_I, test_I,
             f"I: stratified F restricted to multi-timepoint clones ({len(test_I)} clones)")


if __name__ == "__main__":
    main()
