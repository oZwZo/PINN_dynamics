#!/usr/bin/env python
"""Look at Strategy I's test clones; identify NMP/Mast-dominant outliers.

Goal: drop the worst skew-contributing clones one at a time until the test
cell-type distribution at Day 17 matches train within a chosen tolerance.

Outputs:
- Per-clone breakdown of Day-17 fate composition for every Strategy-I test clone.
- A greedy "drop to balance" algorithm: at each step, drop the clone whose
  removal most reduces the max |train_pct - test_pct| across fates.
- Optional: also drop clones from train to push balance further if needed.

Read-only — does not modify the h5ad.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc

PATH = "/rds/user/wz369/hpc-work/PINN_dynamics/data/cordblood_addpop.h5ad"
SEED = 42


def main():
    adata = sc.read_h5ad(PATH)
    clones = adata.obs["clones"].astype(str)
    is_barcoded = clones != "Clone_"

    clone_size = clones[is_barcoded].value_counts()
    multi_cell = set(clone_size[clone_size > 1].index)

    n_tp_per_clone = (
        adata.obs[is_barcoded]
        .groupby(clones[is_barcoded])["timepoint_tx_days"]
        .nunique()
    )
    multi_tp = set(n_tp_per_clone[n_tp_per_clone >= 2].index)

    # Rebuild Strategy I exactly as in _analysis_split_options.py
    d17_mask = adata.obs["timepoint_tx_days"] == 17
    sub = adata.obs[d17_mask & is_barcoded].copy()
    sub["clone"] = sub["clones"].astype(str)
    eligible = multi_tp & multi_cell
    sub_eligible = sub[sub["clone"].isin(eligible)]
    primary = (sub_eligible.groupby("clone")["def_lab"]
                .agg(lambda x: x.value_counts().idxmax()))
    rng = np.random.RandomState(SEED)
    test_I = set()
    for fate, group_clones in primary.groupby(primary):
        n = max(1, int(round(0.10 * len(group_clones))))
        test_I.update(rng.choice(group_clones.index.tolist(), size=n, replace=False).tolist())
    train_I = multi_cell - test_I

    print(f"Strategy I test clones: {len(test_I)}")
    print(f"Strategy I train clones: {len(train_I)}\n")

    # Day-17 fate composition per test clone
    rows = []
    for clone in sorted(test_I):
        clone_d17 = sub[sub["clone"] == clone]
        fates = clone_d17["def_lab"].value_counts()
        total = int(fates.sum())
        if total == 0:
            continue
        # Total cells across all timepoints
        total_all = int(clones[clones == clone].sum() if False else (clones == clone).sum())
        rows.append({
            "clone": clone,
            "d17_cells": total,
            "total_cells": total_all,
            "primary_fate": primary[clone],
            "n_d17_fates": int((fates > 0).sum()),
            "top_fate_pct": float(fates.iloc[0] / total * 100),
        })
    df = pd.DataFrame(rows).sort_values("d17_cells", ascending=False)

    # NMP-dominant and Mast-dominant test clones
    nmp_test_clones = df[df["primary_fate"] == "NMP"].sort_values("d17_cells", ascending=False)
    mast_test_clones = df[df["primary_fate"] == "Mast cell"].sort_values("d17_cells", ascending=False)
    print(f"=== NMP-primary test clones ({len(nmp_test_clones)}) ===")
    print(nmp_test_clones.to_string(index=False))
    print(f"\n=== Mast cell-primary test clones ({len(mast_test_clones)}) ===")
    print(mast_test_clones.to_string(index=False))

    # Compute baseline distribution
    def fate_counts(clones_set):
        m = (clones.isin(clones_set)) & d17_mask
        return adata.obs.loc[m, "def_lab"].value_counts()

    def fate_pcts(s):
        s = s.reindex(adata.obs["def_lab"].cat.categories if hasattr(adata.obs["def_lab"], "cat") else sorted(adata.obs["def_lab"].unique())).fillna(0)
        return (s / max(s.sum(), 1) * 100)

    train_pcts0 = fate_pcts(fate_counts(train_I))
    test_pcts0 = fate_pcts(fate_counts(test_I))
    print(f"\n=== Strategy I baseline ===")
    cmp = pd.DataFrame({"train_pct": train_pcts0.round(1),
                        "test_pct": test_pcts0.round(1),
                        "delta": (test_pcts0 - train_pcts0).round(1)})
    print(cmp.to_string())
    print(f"max |delta| = {cmp['delta'].abs().max():.1f}")

    # ── Greedy drop: at each step, find the clone whose removal most reduces max|delta| ──
    print("\n=== Greedy drop test clones to reduce max|delta| ===")
    cur_test = set(test_I)
    history = []
    for step in range(20):
        candidates = []
        for c in cur_test:
            new_test = cur_test - {c}
            new_pcts = fate_pcts(fate_counts(new_test))
            new_train = fate_pcts(fate_counts(multi_cell - new_test))
            new_delta = (new_pcts - new_train).abs().max()
            candidates.append((c, new_delta, primary[c]))
        candidates.sort(key=lambda x: x[1])
        best_c, best_delta, best_fate = candidates[0]
        # If dropping doesn't improve, stop
        cur_delta = (fate_pcts(fate_counts(cur_test)) - fate_pcts(fate_counts(multi_cell - cur_test))).abs().max()
        if best_delta >= cur_delta:
            print(f"  step {step}: no improvement (cur max|delta|={cur_delta:.1f}). Stop.")
            break
        cur_test.discard(best_c)
        history.append((step, best_c, best_fate, best_delta))
        print(f"  step {step}: dropped {best_c} (primary={best_fate}) → max|delta|={best_delta:.1f}, "
              f"remaining test clones={len(cur_test)}")

    final_train = multi_cell - cur_test
    final_train_pcts = fate_pcts(fate_counts(final_train))
    final_test_pcts = fate_pcts(fate_counts(cur_test))
    final_cmp = pd.DataFrame({"train_pct": final_train_pcts.round(1),
                              "test_pct": final_test_pcts.round(1),
                              "delta": (final_test_pcts - final_train_pcts).round(1)})
    print(f"\n=== Strategy I' (after dropping {len(test_I) - len(cur_test)} clones) ===")
    print(f"Final test clones: {len(cur_test)}")
    print(final_cmp.to_string())
    print(f"max |delta| = {final_cmp['delta'].abs().max():.1f}")

    # Output the kept test clones
    print(f"\nDropped clones ({len(test_I) - len(cur_test)}):")
    for s, c, f, d in history:
        print(f"  {c}  (primary={f})")
    print(f"\nKept test clones ({len(cur_test)}):")
    for c in sorted(cur_test):
        clone_size_n = int((clones == c).sum())
        d17_n = int(sub[sub["clone"] == c].shape[0])
        print(f"  {c}  size={clone_size_n}  d17_cells={d17_n}  primary={primary[c]}")


if __name__ == "__main__":
    main()
