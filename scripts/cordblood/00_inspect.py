#!/usr/bin/env python
"""00_inspect.py — Inspect the CLADES CordBlood_Refine archive.

Writes:
- data/CordBlood/INSPECTION.md (human-readable summary)
- docs/cordblood/findings.md (appended)

Reads:
- data/CordBlood/raw/CordBlood_Refine/adata_update.h5ad
- data/CordBlood/raw/CordBlood_Refine/CordBlood_Obs_Clones.pkl
- data/CordBlood/raw/CordBlood_Refine/data/*.csv
- data/CordBlood/raw/CordBlood_Refine/data/kinetics_array_correction_factor.txt
- data/Complete_LARRY_dataset_adata_preprocessed.h5ad (for comparison)
"""
from __future__ import annotations

import os
import pickle
import textwrap
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path("/rds/user/wz369/hpc-work/PINN_dynamics")
RAW = ROOT / "data/CordBlood/raw/CordBlood_Refine"
OUT_MD = ROOT / "data/CordBlood/INSPECTION.md"
FINDINGS = ROOT / "docs/cordblood/findings.md"
COMP_PATH = ROOT / "data/Complete_LARRY_dataset_adata_preprocessed.h5ad"


def summarize_obs(adata, name="adata"):
    lines = [f"#### {name}.obs ({adata.n_obs} rows × {len(adata.obs.columns)} cols)\n",
             "| column | dtype | n_unique | sample |", "|---|---|---|---|"]
    for c in adata.obs.columns:
        s = adata.obs[c]
        try:
            sample = ", ".join(str(x) for x in pd.unique(s)[:5])
        except Exception:
            sample = "<unhashable>"
        lines.append(f"| `{c}` | {s.dtype} | {s.nunique(dropna=True)} | {sample} |")
    return "\n".join(lines)


def summarize_obsm(adata, name="adata"):
    lines = [f"#### {name}.obsm",
             "| key | shape | dtype |", "|---|---|---|"]
    for k, v in adata.obsm.items():
        lines.append(f"| `{k}` | {tuple(v.shape)} | {v.dtype} |")
    return "\n".join(lines)


def summarize_uns(adata, name="adata"):
    lines = [f"#### {name}.uns",
             "| key | type | shape/keys |", "|---|---|---|"]
    for k, v in adata.uns.items():
        if hasattr(v, "shape"):
            descr = f"shape={tuple(v.shape)}, dtype={v.dtype}"
        elif isinstance(v, dict):
            descr = f"dict, keys={list(v.keys())}"
        elif isinstance(v, (list, tuple)):
            descr = f"{type(v).__name__}, len={len(v)}"
        else:
            descr = f"{type(v).__name__}"
        lines.append(f"| `{k}` | {type(v).__name__} | {descr} |")
    return "\n".join(lines)


def summarize_layers(adata, name="adata"):
    if not adata.layers:
        return f"#### {name}.layers — (empty)"
    lines = [f"#### {name}.layers",
             "| key | shape | dtype |", "|---|---|---|"]
    for k, v in adata.layers.items():
        lines.append(f"| `{k}` | {tuple(v.shape)} | {v.dtype} |")
    return "\n".join(lines)


def detect_norm_state(adata):
    """Best-effort: is X raw counts or log-normalized?"""
    X = adata.X
    if hasattr(X, "toarray"):
        sample = X[:200].toarray()
    else:
        sample = np.asarray(X[:200])
    is_int = np.allclose(sample, sample.astype(int))
    mx = float(sample.max())
    return {
        "looks_like_integer": bool(is_int),
        "max_value": mx,
        "min_value": float(sample.min()),
        "mean_value": float(sample.mean()),
        "guess": "raw_counts" if (is_int and mx > 10) else "log_normalized" if mx < 15 else "unknown",
    }


def cells_by_timepoint_celltype(adata, tp_col, ct_col):
    return pd.crosstab(adata.obs[tp_col], adata.obs[ct_col])


def main():
    print("Loading adata...")
    adata = sc.read_h5ad(RAW / "adata_update.h5ad")
    print(f"  n_obs={adata.n_obs}, n_vars={adata.n_vars}")

    print("Loading clones pkl...")
    with open(RAW / "CordBlood_Obs_Clones.pkl", "rb") as fh:
        clones_df = pickle.load(fh)
    print(f"  pkl shape={clones_df.shape}")

    print("Loading kinetics...")
    kinetics_flat = np.loadtxt(RAW / "data" / "kinetics_array_correction_factor.txt")
    print(f"  kinetics shape={kinetics_flat.shape}")
    # Per the prepare_input.ipynb: shape (13, 3*12) where 13 = 12 meta-clones + 1 total
    kinetics = kinetics_flat.reshape(13, 3, 12)

    # Cell type / population list (from prepare_input.ipynb)
    POPS = ['HSC/MPP 1', 'HSC/MPP 2', 'MEMP', 'Mast cell',
            'Early Erythroid', 'Mid Erythroid', 'Late Erythroid',
            'NMP', 'Mono precur', 'Monocyte', 'DC precursor', 'DC']
    TIMES = [3, 10, 17]

    # ── Comparison with Complete_LARRY ──
    comp_summary = "Not found"
    if COMP_PATH.exists():
        try:
            print("Loading comparison Complete_LARRY...")
            comp = sc.read_h5ad(COMP_PATH, backed="r")
            comp_summary = (
                f"- n_obs={comp.n_obs}, n_vars={comp.n_vars}\n"
                f"- obs columns: {list(comp.obs.columns)}\n"
                f"- obsm keys: {list(comp.obsm.keys())}\n"
                f"- uns keys: {list(comp.uns.keys())}\n"
                f"- Same cell count as CLADES? {comp.n_obs == adata.n_obs}\n"
            )
            comp.file.close()
        except Exception as e:
            comp_summary = f"Failed to load: {e}"

    # ── Crosstabs ──
    timepoint_col = "Timepoint"  # confirmed from notebook
    celltype_col = "def_lab"     # confirmed
    clone_col = "clones"

    crosstab_tp_ct = cells_by_timepoint_celltype(adata, timepoint_col, celltype_col)
    crosstab_clone_tp = pd.crosstab(adata.obs[clone_col], adata.obs[timepoint_col])

    # Clones with barcode (not "Clone_") + clone size distribution
    is_barcoded = adata.obs[clone_col] != "Clone_"
    n_barcoded = int(is_barcoded.sum())
    clone_sizes = adata.obs[is_barcoded][clone_col].value_counts()
    n_singletons = int((clone_sizes == 1).sum())
    n_multicell_clones = int((clone_sizes > 1).sum())

    # Clones spanning multiple timepoints
    clone_n_tp = (crosstab_clone_tp > 0).sum(axis=1)
    clones_multi_tp = int((clone_n_tp > 1).sum())
    clones_single_tp = int((clone_n_tp == 1).sum())

    # Day17 specific: how many barcoded cells at the latest timepoint
    day17_cells = adata.obs[adata.obs[timepoint_col] == "Day17"]
    day17_barcoded = day17_cells[day17_cells[clone_col] != "Clone_"]

    # Norm state of X
    norm = detect_norm_state(adata)

    # ── Population dynamics from kinetics ──
    # Total cell count per timepoint = kinetics.sum over leiden + population
    # Per the notebook, kinetics[-1, idj, idk] is the "all" row containing total adata cell count
    total_per_tp = kinetics[-1].sum(axis=1)  # shape (3,)
    metaclone_per_tp = kinetics[:-1].sum(axis=2)  # shape (12, 3): per-metaclone per-tp totals
    pop_per_tp_pop = kinetics[-1]  # shape (3, 12): per-population, per-timepoint

    # ── Write INSPECTION.md ──
    md = []
    md.append("# CLADES CordBlood Inspection\n")
    md.append("Source archive: `data/CordBlood/raw/CordBlood_Refine/` "
              f"(downloaded {pd.Timestamp.now().isoformat()})\n")
    md.append(f"Generated by `{__file__}`.\n")

    md.append("## 1. AnnData shape\n")
    md.append(f"- `n_obs` = {adata.n_obs:,}\n- `n_vars` = {adata.n_vars:,}\n")

    md.append("## 2. Normalization state of `adata.X`\n")
    for k, v in norm.items():
        md.append(f"- {k}: `{v}`\n")
    md.append(f"\n**Best guess: `{norm['guess']}`** "
              "(if `log_normalized` we skip the normalize+log1p step; "
              "if `raw_counts` we run sc.pp.normalize_total + sc.pp.log1p in Phase C.)\n")

    md.append("\n## 3. obs / obsm / uns / layers\n")
    md.append(summarize_obs(adata) + "\n")
    md.append(summarize_obsm(adata) + "\n")
    md.append(summarize_uns(adata) + "\n")
    md.append(summarize_layers(adata) + "\n")

    md.append("\n## 4. Time × cell-type crosstab\n")
    md.append("```\n" + crosstab_tp_ct.to_string() + "\n```\n")
    md.append(f"\n**Total cells per timepoint** (from adata): "
              f"{dict(adata.obs[timepoint_col].value_counts())}\n")

    md.append("\n## 5. Clone barcodes\n")
    md.append(f"- Unique values in `obs['{clone_col}']` (incl. unbarcoded sentinel): "
              f"{adata.obs[clone_col].nunique()}\n")
    md.append(f"- Unbarcoded cells (label == `Clone_`): {int((~is_barcoded).sum()):,}\n")
    md.append(f"- Barcoded cells: {n_barcoded:,}\n")
    md.append(f"- Unique clone IDs (excluding sentinel): "
              f"{adata.obs[is_barcoded][clone_col].nunique():,}\n")
    md.append(f"- Singleton clones (n_cells == 1): {n_singletons:,}\n")
    md.append(f"- Multi-cell clones (n_cells > 1): {n_multicell_clones:,}\n")
    md.append(f"- Clones spanning >1 timepoint: {clones_multi_tp:,}\n")
    md.append(f"- Clones at exactly 1 timepoint: {clones_single_tp:,}\n")
    md.append(f"- Day17 cells (all): {day17_cells.shape[0]:,}\n")
    md.append(f"- Day17 cells (barcoded): {day17_barcoded.shape[0]:,}\n")

    md.append("\n### Clone size distribution (multi-cell only)\n")
    bins = [1, 2, 3, 5, 10, 20, 50, 100, 1_000_000]
    cuts = pd.cut(clone_sizes, bins=bins, right=False).value_counts().sort_index()
    md.append("```\n" + cuts.to_string() + "\n```\n")

    md.append("\n## 6. CordBlood_Obs_Clones.pkl\n")
    md.append(f"- DataFrame shape: {clones_df.shape}\n")
    md.append(f"- First 14 columns (info): {list(clones_df.columns[:14])}\n")
    md.append(f"- Remaining columns: {clones_df.shape[1] - 14} (clone one-hot)\n")
    md.append(f"- Sample clone columns: {list(clones_df.columns[14:19])} ...\n")
    md.append(f"- Index name: `{clones_df.index.name}`; first 3 indices: {list(clones_df.index[:3])}\n")
    md.append(f"- All adata.obs_names present in clones pkl? "
              f"{set(adata.obs_names).issubset(set(clones_df.index))}\n")

    md.append("\n## 7. Population dynamics (`kinetics_array_correction_factor.txt`)\n")
    md.append(f"- Shape (flat): {kinetics_flat.shape}\n")
    md.append(f"- Reshaped: (13 meta-clones+1total, 3 timepoints, 12 populations)\n")
    md.append("\n### Total scaled cell count per timepoint (last row sum)\n")
    md.append("```\n" + str(dict(zip(TIMES, total_per_tp.tolist()))) + "\n```\n")
    md.append("\n### Per-population scaled count per timepoint (last row of kinetics)\n")
    pop_tp_df = pd.DataFrame(pop_per_tp_pop, index=[f"Day{t}" for t in TIMES], columns=POPS)
    md.append("```\n" + pop_tp_df.to_string() + "\n```\n")
    md.append("\n**Interpretation:** `kinetics[-1, t, :]` is the total expanded cell count "
              "at each timepoint after applying FACS correction factors (Day3 ×4.91, "
              "Day10 ×530.6, Day17 ×508.7). This becomes `uns['pop']['mean']`.\n")

    md.append("\n## 8. Comparison vs Complete_LARRY_dataset_adata_preprocessed.h5ad\n")
    md.append(comp_summary + "\n")

    md.append("\n## 9. Decisions (stop-and-decide gate)\n")
    md.append(f"- **Time column:** `{timepoint_col}` (str: Day3, Day10, Day17). "
              "Will cast to int(3,10,17) and save to `obs['timepoint_tx_days']`.\n")
    md.append(f"- **Clone column:** `{clone_col}` (`Clone_<N>` strings; "
              f"sentinel `Clone_` marks unbarcoded). Use only barcoded cells for fate eval.\n")
    md.append(f"- **Cell-type column for fate eval:** `{celltype_col}` (12 populations).\n")
    md.append(f"- **PCA source:** `X_pca` present in obsm. `X_pca_harmony` **NOT** present. "
              "Will use `X_pca` (no harmony correction available).\n")
    md.append(f"- **DM:** `X_diffmap` present (scanpy version). Will re-compute via palantir "
              "for `DM_EigenVectors` (+ multiscaled).\n")
    md.append(f"- **F_obs timepoint:** **Day17** (latest, most cells, most clones with "
              "fates resolved). This mirrors the Klein 'day 6' convention.\n")
    md.append(f"- **uns['pop'] source:** kinetics_array_correction_factor.txt → "
              "uns['pop'] = {{'t': [3,10,17], 'mean': total_per_tp, "
              "'std': sqrt(mean) (Poisson fallback unless we find better), "
              "'variance_source': 'kinetics_total'}}.\n")

    OUT_MD.write_text("".join(md))
    print(f"Wrote {OUT_MD}")

    # Also append a brief summary to findings.md
    summary = textwrap.dedent(f"""
        ## Phase A inspection (auto-generated {pd.Timestamp.now().isoformat()})

        - **AnnData:** {adata.n_obs:,} cells × {adata.n_vars:,} genes
        - **Timepoints:** Day3 ({(adata.obs[timepoint_col] == 'Day3').sum():,}),
          Day10 ({(adata.obs[timepoint_col] == 'Day10').sum():,}),
          Day17 ({(adata.obs[timepoint_col] == 'Day17').sum():,})
        - **Cell types (`def_lab`):** {adata.obs[celltype_col].nunique()} categories
        - **Clones:** {n_barcoded:,} barcoded cells across
          {adata.obs[is_barcoded][clone_col].nunique():,} unique clone IDs
          ({n_singletons:,} singletons, {n_multicell_clones:,} multi-cell)
        - **X normalization:** {norm['guess']} (max={norm['max_value']:.2f}, integer={norm['looks_like_integer']})
        - **PCA source:** `X_pca` (no harmony available)
        - **F_obs timepoint:** Day17
        - **uns['pop'] source:** kinetics_array_correction_factor.txt (totals per timepoint)
        - Full report: `data/CordBlood/INSPECTION.md`
    """).strip() + "\n"

    with open(FINDINGS, "a") as fh:
        fh.write("\n" + summary)
    print(f"Appended summary to {FINDINGS}")


if __name__ == "__main__":
    main()
