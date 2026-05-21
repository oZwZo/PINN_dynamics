#!/usr/bin/env python
"""
Preprocess the synthetic 5-D Fokker–Planck h5ad so it satisfies every field
that pseudodynamics' `TwoTimpepoint_AnnDS` reader (via `main_train.py`) reads.

Source file: data/synthesized_data/5Dim_ncs_synparam_Jan23/syn_cell_0_4.h5ad
Already present and OK:
    obsm['X_data']   (22000, 5) float32   — cellstate
    obsm['Delta_DM'] (22000, 5) float32   — v-attachment
    uns['pop']['t']     (11,)  float32    — timepoints
    uns['pop']['mean']  (11,)  float32    — population at each timepoint

Missing / wrong, added by this script:
    obs['timepoint_tx_days']         — main_train.py / reader hard-codes this
                                       obs key for timepoint. Cast to float and
                                       snapped to popD['t'] so the
                                       `obs[k] == t` masks in reader.py:107
                                       match exactly (current obs.time is
                                       float64 with 1e-16 rounding artifacts).
    uns['pop']['std']                — used by reader.py:123 as
                                       `var_ls = std**2 / n_lib`. With a
                                       deterministic synthesis the true std is
                                       0; we set 5 % of the per-timepoint mean
                                       so the GNLL-style loss is well-defined
                                       without dominating training.
    uns['pop']['var']                — used at _base_Dataset.py:59 in
                                       `popD['var'] = popD['var'] / N0`. The
                                       field is conceptually the same as
                                       'std' (both are squared downstream),
                                       so we mirror it.
    uns['pop']['n_lib']              — used at reader.py:123 / _base:155.
                                       Synthetic dataset = 1 deterministic run.
"""
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc


REPO_ROOT = Path("/rds/user/wz369/hpc-work/pseudodynamics_plus")
SRC = REPO_ROOT / "data/synthesized_data/5Dim_ncs_synparam_Jan23/syn_cell_0_4.h5ad"
DST = REPO_ROOT / "data/synthetic_FP_5D.h5ad"

# 5 % relative noise floor; matches the magnitude of synthesis noise but is
# small enough not to swamp the PDE residual term.
STD_REL_FLOOR = 0.05


def prepare(src: Path, dst: Path) -> None:
    print(f"reading {src}")
    a = sc.read_h5ad(src)
    print(f"  shape={a.shape}  obs cols={list(a.obs.columns)}  obsm={list(a.obsm)}")

    pop_t = np.asarray(a.uns["pop"]["t"], dtype=np.float64)
    pop_mean = np.asarray(a.uns["pop"]["mean"], dtype=np.float64)
    assert pop_t.shape == pop_mean.shape, (pop_t.shape, pop_mean.shape)

    # 1. obs['timepoint_tx_days']
    # Snap obs.time to the nearest pop['t'] value to eliminate
    # float64-vs-float32 rounding (== comparisons in the reader).
    raw_time = np.asarray(a.obs["time"].values, dtype=np.float64)
    tp_idx = np.argmin(np.abs(raw_time[:, None] - pop_t[None, :]), axis=1)
    snapped = pop_t[tp_idx]
    drift = np.max(np.abs(snapped - raw_time))
    assert drift < 1e-3, f"unexpected drift {drift} between obs.time and pop.t"
    a.obs["timepoint_tx_days"] = snapped.astype(np.float64)

    # Sanity: cell counts per timepoint, must be positive everywhere.
    counts = np.array([(a.obs["timepoint_tx_days"].values == t).sum() for t in pop_t])
    assert (counts > 0).all(), f"empty timepoint(s): counts={counts.tolist()}"
    print(f"  timepoint_tx_days snapped; cells/timepoint = {counts.tolist()}")

    # 2. uns['pop'] augmentation. Convert to a clean float64 dict.
    new_pop = {
        "t":     pop_t.astype(np.float64),
        "mean":  pop_mean.astype(np.float64),
        "std":   (STD_REL_FLOOR * pop_mean).astype(np.float64),
        "var":   (STD_REL_FLOOR * pop_mean).astype(np.float64),  # field semantics: std (squared downstream)
        "n_lib": np.ones_like(pop_t, dtype=np.float64),
    }
    a.uns["pop"] = new_pop
    print(f"  pop keys -> {list(new_pop)}; std = {STD_REL_FLOOR} × mean")

    # 3. Write
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    a.write_h5ad(dst)
    print(f"wrote {dst}  ({dst.stat().st_size / 1e6:.1f} MB)")


def verify(dst: Path) -> None:
    """Round-trip read and assert every field the reader will touch."""
    a = sc.read_h5ad(dst)
    pop = a.uns["pop"]
    for k in ("t", "mean", "std", "var", "n_lib"):
        assert k in pop, f"pop missing {k!r}"
    assert "timepoint_tx_days" in a.obs, list(a.obs.columns)
    assert "X_data" in a.obsm and a.obsm["X_data"].shape == (a.n_obs, 5)
    assert "Delta_DM" in a.obsm and a.obsm["Delta_DM"].shape == (a.n_obs, 5)
    assert np.isfinite(np.asarray(a.obsm["Delta_DM"])).all(), "Delta_DM has nan/inf"

    pop_t = np.asarray(pop["t"])
    obs_t = np.asarray(a.obs["timepoint_tx_days"].values)
    # every timepoint in pop['t'] must have at least one matching cell
    for t in pop_t:
        assert (obs_t == t).any(), f"no cell with timepoint_tx_days == {t}"
    # variance basis is positive
    assert (np.asarray(pop["std"]) > 0).all(), "pop['std'] has non-positive entries"
    print(f"verify OK  shape={a.shape}  n_t={len(pop_t)}  pop[t]={pop_t.tolist()}")


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"source h5ad missing: {SRC}")
    prepare(SRC, DST)
    verify(DST)


if __name__ == "__main__":
    main()
