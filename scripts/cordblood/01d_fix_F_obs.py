#!/usr/bin/env python
"""01d_fix_F_obs.py — Regenerate F_obs.csv with Klein-style semantics.

Klein convention:
  F_obs.index   = cell IDs at the START timepoint (the cells we forward-simulate FROM)
  F_obs.columns = fate cell types
  F_obs values  = one-hot, where the 1 is the dominant fate of THAT cell's clone
                  at the END timepoint (Day 17 here).

This script overwrites:
  data/CordBlood/F_obs_day17.csv        (start cells at Day 3)
  data/CordBlood/clone_proportions_day17.csv  (per-clone fate at Day 17; unchanged)

Cells in F_obs:
  - At Day 3 (tp_first)
  - Their clone has ≥1 cell at Day 17 (so a fate is observable)
  - The clone is in `clone_proportions_day17.csv` (multi-cell barcoded)

The old F_obs_day17.csv listed Day-17 cells themselves; that's useful for some
analyses but is NOT what `run_fate_evaluation` expects.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path("/rds/user/wz369/hpc-work/PINN_dynamics")
ADATA_PATH = ROOT / "data" / "cordblood_addpop.h5ad"
F_OBS_OUT = ROOT / "data" / "CordBlood" / "F_obs_day17.csv"
CLONE_PROP_OUT = ROOT / "data" / "CordBlood" / "clone_proportions_day17.csv"

START_TP = 3
FATE_TP = 17


def main():
    print(f"Loading {ADATA_PATH} ...")
    adata = sc.read_h5ad(ADATA_PATH)
    obs = adata.obs

    clones = obs["clones"].astype(str)
    is_barcoded = clones != "Clone_"

    fate_cats = (
        list(obs["def_lab"].cat.categories)
        if hasattr(obs["def_lab"], "cat")
        else sorted(obs["def_lab"].unique())
    )

    # Per-clone fate counts at Day 17
    d17_mask = (obs["timepoint_tx_days"] == FATE_TP) & is_barcoded
    d17 = obs.loc[d17_mask].copy()
    d17["clone"] = clones[d17_mask]

    clone_fate_counts = (
        d17.groupby("clone")["def_lab"]
            .value_counts()
            .unstack(fill_value=0)
            .reindex(columns=fate_cats, fill_value=0)
    )
    clone_proportions = clone_fate_counts.div(clone_fate_counts.sum(axis=1), axis=0)
    # Dominant fate per clone (the "ground-truth" label assigned to all its Day-3 cells)
    clone_dominant_fate = clone_fate_counts.idxmax(axis=1)
    # Restrict to clones with >=2 Day-17 cells (need at least some fate signal)
    clones_with_d17 = set(clone_fate_counts.index[clone_fate_counts.sum(axis=1) >= 2])
    print(f"  Clones with >=2 Day-17 cells: {len(clones_with_d17):,}")

    # Day-3 cells in those clones
    d3_mask = (obs["timepoint_tx_days"] == START_TP) & is_barcoded & clones.isin(clones_with_d17)
    d3_cells = obs.index[d3_mask]
    d3_clones = clones[d3_mask]
    print(f"  Day-3 cells eligible (barcoded + clone has Day-17 fate): {len(d3_cells):,}")

    # Build one-hot F_obs at Day 3
    F_obs = pd.DataFrame(0, index=d3_cells, columns=fate_cats, dtype=int)
    for cb, cl in zip(d3_cells, d3_clones):
        F_obs.loc[cb, clone_dominant_fate[cl]] = 1

    F_OBS_OUT.parent.mkdir(parents=True, exist_ok=True)
    F_obs.to_csv(F_OBS_OUT)
    print(f"  Wrote {F_OBS_OUT}  shape={F_obs.shape}")

    # Clone proportions (per-clone Day-17 fractions over all clones with Day-17 cells)
    clone_proportions.to_csv(CLONE_PROP_OUT)
    print(f"  Wrote {CLONE_PROP_OUT}  shape={clone_proportions.shape}")

    # Quick summary
    print("\n  Per-fate Day-3 cell counts (F_obs row sums per column):")
    print(F_obs.sum().to_string())
    print(f"\n  Total Day-3 cells in F_obs: {F_obs.shape[0]:,}")
    print(f"  Unique clones spanned: {d3_clones.nunique():,}")


if __name__ == "__main__":
    main()
