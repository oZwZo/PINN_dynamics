"""Schema validator for cord blood AnnData files.

Used by:
- scripts/cordblood/01_preprocess.py (at the end, before writing)
- scripts/cordblood/dispatch_task.sh (every training task confirms its inputs)

Fails loud (raises) on any missing or wrong-shape key.
"""
from __future__ import annotations

import numpy as np


# Mandatory schema for `data/cordblood_addpop.h5ad`.
# Shapes use `n` for n_obs; the validator enforces shape[0] == n_obs.
REQUIRED_OBS = {
    "timepoint_tx_days": "integer-like",
    "Timepoint": "category",
    "clones": "category",
    "def_lab": "category",
    "split": "string",  # 'train' / 'val' / 'test' / 'train_unbarcoded'
    "Well": "integer-like",  # compat column: 2 = test, 0 = otherwise (legacy Klein contract)
}

REQUIRED_OBSM = {
    "X_pca": (50,),                    # 50 PCs preserved as-is from CLADES
    "X_pca_scaled": (30,),             # z-scored (fit on train), 30 dims
    "DM_EigenVectors": None,           # palantir; n_dims set by uns['dm_n_dims']
    "DM_EigenVectors_scaled": None,    # z-scored (fit on train), same n_dims
    "DM_EigenVectors_multiscaled": None,
    "delta_PC": (30,),                 # 20-repeat mean of pdp.tl.sample_deltax
    "delta_DM": None,                  # same n_dims as DM_EigenVectors
}

REQUIRED_UNS_KEYS = {
    "pca_scaler",
    "dm_scaler",
    "pca_source",          # 'X_pca' or 'X_pca_harmony'
    "dm_n_dims",           # int — number of DM dims palantir kept
    "pop",                 # {'t': arr, 'mean': arr, 'std': arr, 'var': arr,
                           #  'variance_source': str, 'n_lib': arr_or_None}
}

REQUIRED_SCALER_FIELDS = {"mean", "std", "n_dims", "source_key", "fit_on", "n_train_cells"}
REQUIRED_POP_FIELDS = {"t", "mean", "std", "var"}


def validate(adata, *, strict_extras: bool = False):
    """Validate cord blood AnnData. Raises ValueError on any mismatch.

    Parameters
    ----------
    adata : AnnData
    strict_extras : bool
        If True, also fail when EXTRA obs/obsm/uns keys are present beyond the schema.
        Default False (extras are common from upstream and harmless).
    """
    errors = []

    # obs
    for col, kind in REQUIRED_OBS.items():
        if col not in adata.obs.columns:
            errors.append(f"obs['{col}'] missing")
            continue
        if kind == "integer-like":
            try:
                _ = adata.obs[col].astype(int)
            except Exception as e:
                errors.append(f"obs['{col}'] not castable to int: {e}")

    # split values
    if "split" in adata.obs.columns:
        seen = set(adata.obs["split"].astype(str).unique())
        allowed = {"train", "val", "test", "train_unbarcoded"}
        unexpected = seen - allowed
        if unexpected:
            errors.append(f"obs['split'] has unexpected values: {sorted(unexpected)}")
        if "train" not in seen:
            errors.append("obs['split'] has no 'train' cells")
        if "test" not in seen:
            errors.append("obs['split'] has no 'test' cells")

    # obsm
    for key, shape_tail in REQUIRED_OBSM.items():
        if key not in adata.obsm:
            errors.append(f"obsm['{key}'] missing")
            continue
        arr = adata.obsm[key]
        if arr.shape[0] != adata.n_obs:
            errors.append(f"obsm['{key}'] shape[0]={arr.shape[0]} != n_obs={adata.n_obs}")
        if shape_tail is not None and arr.shape[1:] != shape_tail:
            errors.append(f"obsm['{key}'] expected shape (n_obs, {shape_tail[0]}), got {arr.shape}")
        if np.isnan(arr).any():
            errors.append(f"obsm['{key}'] contains NaN")

    # uns
    for key in REQUIRED_UNS_KEYS:
        if key not in adata.uns:
            errors.append(f"uns['{key}'] missing")

    # uns scalers
    for sk in ("pca_scaler", "dm_scaler"):
        if sk in adata.uns:
            scaler = adata.uns[sk]
            if not isinstance(scaler, dict):
                errors.append(f"uns['{sk}'] must be dict, got {type(scaler).__name__}")
                continue
            missing = REQUIRED_SCALER_FIELDS - set(scaler.keys())
            if missing:
                errors.append(f"uns['{sk}'] missing keys: {sorted(missing)}")
            if scaler.get("fit_on") != "train":
                errors.append(f"uns['{sk}']['fit_on']={scaler.get('fit_on')!r}, expected 'train'")
            mean = np.asarray(scaler["mean"])
            std = np.asarray(scaler["std"])
            if mean.shape != std.shape:
                errors.append(f"uns['{sk}'] mean.shape={mean.shape} != std.shape={std.shape}")
            if (std < 1e-6).any():
                errors.append(f"uns['{sk}'] std has zero-variance dims: "
                              f"{np.where(std < 1e-6)[0].tolist()}")

    # uns pop
    if "pop" in adata.uns:
        pop = adata.uns["pop"]
        if not isinstance(pop, dict):
            errors.append(f"uns['pop'] must be dict, got {type(pop).__name__}")
        else:
            missing = REQUIRED_POP_FIELDS - set(pop.keys())
            if missing:
                errors.append(f"uns['pop'] missing keys: {sorted(missing)}")
            t = np.asarray(pop.get("t", []))
            mean = np.asarray(pop.get("mean", []))
            if t.shape != mean.shape:
                errors.append(f"uns['pop'] t.shape={t.shape} != mean.shape={mean.shape}")

    if strict_extras:
        unexpected_obsm = set(adata.obsm.keys()) - set(REQUIRED_OBSM.keys()) - {
            # known harmless extras from upstream:
            "X_diffmap", "X_umap",
        }
        if unexpected_obsm:
            errors.append(f"unexpected obsm keys: {sorted(unexpected_obsm)}")

    if errors:
        raise ValueError(
            "cordblood AnnData schema validation FAILED:\n  - "
            + "\n  - ".join(errors)
        )

    return True


if __name__ == "__main__":
    import sys
    import scanpy as sc

    path = sys.argv[1] if len(sys.argv) > 1 else "data/cordblood_addpop.h5ad"
    adata = sc.read_h5ad(path)
    validate(adata)
    print(f"OK  {path}  ({adata.n_obs} cells x {adata.n_vars} genes)")
