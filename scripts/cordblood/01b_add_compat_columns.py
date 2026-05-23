#!/usr/bin/env python
"""01b_add_compat_columns.py — Add legacy-Klein compat columns to cordblood_addpop.h5ad.

Some downstream eval scripts (OT-CFM, SF2M, TIGON, PRESCIENT 03_evaluate.py)
hard-code `adata.obs.Well == 2` to mask the test set, with no --well_col flag.
Patching all four is more disruptive than adding a single compat column.

This script:
  - Adds `obs['Well']` (int): 2 for test, 0 for everything else.
Idempotent — running on an already-patched file is a no-op.

Note: starting from the next preprocessing run, `01_preprocess.py` adds the
Well column directly so this post-processor is unnecessary. It's kept for
existing h5ad files that pre-date the inline patch.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import scanpy as sc

ROOT = Path("/rds/user/wz369/hpc-work/PINN_dynamics")
DEFAULT_ADATA = ROOT / "data" / "cordblood_addpop.h5ad"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_path", default=str(DEFAULT_ADATA))
    args = p.parse_args()

    path = Path(args.data_path)
    print(f"Loading {path} ...")
    adata = sc.read_h5ad(path)

    wrote = False
    # ── Well (int): 2=test, 0=otherwise (legacy Klein contract) ──
    if "Well" not in adata.obs.columns:
        print("  Adding obs['Well'] from obs['split'] ...")
        adata.obs["Well"] = np.where(adata.obs["split"].astype(str) == "test", 2, 0).astype(int)
        wrote = True
    else:
        derived = np.where(adata.obs["split"].astype(str) == "test", 2, 0).astype(int)
        if int((derived != adata.obs["Well"].astype(int).values).sum()):
            adata.obs["Well"] = derived
            wrote = True

    # ── Annotation (Klein cell-type column name). Cord blood ground truth = def_lab. ──
    if "Annotation" not in adata.obs.columns:
        print("  Adding obs['Annotation'] = obs['def_lab'] ...")
        adata.obs["Annotation"] = adata.obs["def_lab"]
        wrote = True

    # ── label_man (Klein fine-grained cell-type column). No fine-grained in cord blood; alias to def_lab. ──
    if "label_man" not in adata.obs.columns:
        print("  Adding obs['label_man'] = obs['def_lab'] (no separate fine-grained labels in CLADES) ...")
        adata.obs["label_man"] = adata.obs["def_lab"]
        wrote = True

    # ── Time_point (alt time key used by MIOFlow scripts). ──
    if "Time_point" not in adata.obs.columns:
        print("  Adding obs['Time_point'] = obs['Timepoint'] ...")
        adata.obs["Time_point"] = adata.obs["Timepoint"]
        wrote = True

    # ── PC_scaler / DM_scaler uppercase uns aliases (legacy Klein scripts read these). ──
    # otcfm/03_evaluate.py and sf2m/03_evaluate.py hardcode `adata.uns['PC_scaler']`
    # and `adata.uns['DM_scaler']`. Mirror those from our `pca_scaler`/`dm_scaler`.
    if "pca_scaler" in adata.uns and "PC_scaler" not in adata.uns:
        print("  Adding uns['PC_scaler'] = uns['pca_scaler'] ...")
        adata.uns["PC_scaler"] = adata.uns["pca_scaler"]
        wrote = True
    if "dm_scaler" in adata.uns and "DM_scaler" not in adata.uns:
        print("  Adding uns['DM_scaler'] = uns['dm_scaler'] ...")
        adata.uns["DM_scaler"] = adata.uns["dm_scaler"]
        wrote = True

    counts = {int(k): int(v) for k, v in adata.obs["Well"].value_counts().items()}
    print(f"  obs['Well'] counts: {counts}")
    print(f"  obs columns now: {sorted(adata.obs.columns)}")
    if not wrote:
        print("  Nothing changed — adata already had all compat columns.")
        return
    print(f"Writing {path} ...")
    adata.write_h5ad(path)
    print("Done.")


if __name__ == "__main__":
    main()
