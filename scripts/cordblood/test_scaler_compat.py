"""Test compat for the refactored compute_scaler_from_adata.

Covers:
1. Klein backward-compat — no uns scaler → on-the-fly path returns mean/std
   identical to the legacy logic (`raw.mean/std` with clip>=1e-6).
2. Cord blood path — uns['pca_scaler'] present → returned verbatim.
3. n_dims truncation — both paths respect the dims argument.

Run:
    python scripts/cordblood/test_scaler_compat.py
"""
import sys
from pathlib import Path

import numpy as np
import anndata as ad

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from fate_eval_pipeline import compute_scaler_from_adata


def _make_synth_adata(*, n=1000, d=50, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 5)).astype(np.float32)
    obs = {"cell_id": np.arange(n).astype(str)}
    adata = ad.AnnData(X=X, obs=obs)
    adata.obsm["X_pca"] = rng.normal(loc=3.14, scale=2.0, size=(n, d)).astype(np.float64)
    adata.obsm["DM_EigenVectors"] = rng.normal(loc=-0.5, scale=0.1, size=(n, 10)).astype(np.float64)
    return adata


def test_klein_path_pca():
    """No uns scaler → on-the-fly fit. Must match legacy mean/std exactly."""
    adata = _make_synth_adata()
    n_dims = 30
    out = compute_scaler_from_adata(adata, "X_pca_scaled", n_dims)
    raw = adata.obsm["X_pca"][:, :n_dims].astype(np.float64)
    np.testing.assert_array_equal(out["mean"], raw.mean(axis=0))
    np.testing.assert_array_equal(out["std"], np.clip(raw.std(axis=0), 1e-6, None))
    assert out["mean"].shape == (n_dims,)
    assert out["std"].shape == (n_dims,)
    print("PASS  Klein path (X_pca_scaled) — on-the-fly fit matches legacy")


def test_klein_path_dm():
    adata = _make_synth_adata()
    n_dims = 10
    out = compute_scaler_from_adata(adata, "DM_EigenVectors_scaled", n_dims)
    raw = adata.obsm["DM_EigenVectors"][:, :n_dims].astype(np.float64)
    np.testing.assert_array_equal(out["mean"], raw.mean(axis=0))
    np.testing.assert_array_equal(out["std"], np.clip(raw.std(axis=0), 1e-6, None))
    print("PASS  Klein path (DM_EigenVectors_scaled)")


def test_cordblood_path_pca():
    """uns['pca_scaler'] present → returned verbatim, regardless of obsm content."""
    adata = _make_synth_adata()
    n_dims = 30
    # Inject a custom scaler that differs from raw stats
    custom_mean = np.linspace(-1.0, 1.0, 30).astype(np.float64)
    custom_std = np.ones(30, dtype=np.float64) * 0.7
    adata.uns["pca_scaler"] = {
        "mean": custom_mean,
        "std": custom_std,
        "n_dims": 30,
        "source_key": "X_pca",
        "fit_on": "train",
        "n_train_cells": 800,
    }
    out = compute_scaler_from_adata(adata, "X_pca_scaled", n_dims)
    np.testing.assert_array_equal(out["mean"], custom_mean[:n_dims])
    np.testing.assert_array_equal(out["std"], custom_std[:n_dims])
    print("PASS  Cord-blood path (X_pca_scaled) — uns scaler returned verbatim")


def test_cordblood_path_pca_harmony():
    """X_pca_harmony_scaled also routes to uns['pca_scaler']."""
    adata = _make_synth_adata()
    n_dims = 30
    custom_mean = np.full(30, 2.71828, dtype=np.float64)
    custom_std = np.full(30, 0.5, dtype=np.float64)
    adata.uns["pca_scaler"] = {
        "mean": custom_mean, "std": custom_std,
        "n_dims": 30, "source_key": "X_pca_harmony",
        "fit_on": "train", "n_train_cells": 1,
    }
    out = compute_scaler_from_adata(adata, "X_pca_harmony_scaled", n_dims)
    np.testing.assert_array_equal(out["mean"], custom_mean)
    np.testing.assert_array_equal(out["std"], custom_std)
    print("PASS  Cord-blood path (X_pca_harmony_scaled)")


def test_cordblood_path_dm():
    adata = _make_synth_adata()
    n_dims = 8
    custom_mean = np.arange(10, dtype=np.float64) * 0.1
    custom_std = np.arange(10, dtype=np.float64) * 0.01 + 1.0
    adata.uns["dm_scaler"] = {
        "mean": custom_mean, "std": custom_std,
        "n_dims": 10, "source_key": "DM_EigenVectors",
        "fit_on": "train", "n_train_cells": 100,
    }
    out = compute_scaler_from_adata(adata, "DM_EigenVectors_scaled", n_dims)
    # Truncation to n_dims=8 must apply
    np.testing.assert_array_equal(out["mean"], custom_mean[:8])
    np.testing.assert_array_equal(out["std"], custom_std[:8])
    print("PASS  Cord-blood path (DM_EigenVectors_scaled) — n_dims truncation")


def test_non_scaled_key_returns_none():
    adata = _make_synth_adata()
    out = compute_scaler_from_adata(adata, "X_pca", 30)
    assert out is None
    print("PASS  Non-scaled key returns None")


def test_missing_raw_key_raises():
    """If both paths fail (no uns scaler, missing raw obsm), must raise."""
    adata = _make_synth_adata()
    del adata.obsm["X_pca"]  # remove raw key so fallback path errors
    try:
        compute_scaler_from_adata(adata, "X_pca_scaled", 30)
    except KeyError as e:
        print(f"PASS  Missing raw key raises KeyError: {e}")
        return
    raise AssertionError("Expected KeyError but none raised")


def test_uns_path_does_not_need_raw_obsm():
    """If uns scaler is present, raw obsm absence is OK."""
    adata = _make_synth_adata()
    custom_mean = np.zeros(30, dtype=np.float64)
    custom_std = np.ones(30, dtype=np.float64)
    adata.uns["pca_scaler"] = {
        "mean": custom_mean, "std": custom_std,
        "n_dims": 30, "source_key": "X_pca",
        "fit_on": "train", "n_train_cells": 1,
    }
    del adata.obsm["X_pca"]
    out = compute_scaler_from_adata(adata, "X_pca_scaled", 30)
    np.testing.assert_array_equal(out["mean"], custom_mean)
    np.testing.assert_array_equal(out["std"], custom_std)
    print("PASS  uns path does not need raw obsm")


def test_uns_scaler_smaller_than_n_dims_raises():
    """If uns scaler has fewer dims than requested n_dims, must raise (not silently truncate)."""
    adata = _make_synth_adata()
    # Scaler stored with only 10 dims
    adata.uns["pca_scaler"] = {
        "mean": np.zeros(10, dtype=np.float64),
        "std":  np.ones(10, dtype=np.float64),
        "n_dims": 10, "source_key": "X_pca",
        "fit_on": "train", "n_train_cells": 1,
    }
    try:
        compute_scaler_from_adata(adata, "X_pca_scaled", 30)
    except ValueError as e:
        assert "only 10 dims" in str(e) and "n_dims=30" in str(e), f"unexpected msg: {e}"
        print(f"PASS  uns scaler smaller than n_dims raises ValueError")
        return
    raise AssertionError("Expected ValueError for under-dimensioned uns scaler")


def test_uns_path_matches_on_the_fly_when_stats_match():
    """If uns scaler stores the EXACT same mean/std as raw obsm fit, output must match."""
    adata = _make_synth_adata()
    # Compute the on-the-fly answer first (fallback path)
    out_fallback = compute_scaler_from_adata(adata, "X_pca_scaled", 30)
    # Now stash those same stats into uns and verify uns path returns identical
    adata.uns["pca_scaler"] = {
        "mean": out_fallback["mean"].copy(),
        "std": out_fallback["std"].copy(),
        "n_dims": 30, "source_key": "X_pca",
        "fit_on": "train", "n_train_cells": adata.n_obs,
    }
    out_uns = compute_scaler_from_adata(adata, "X_pca_scaled", 30)
    np.testing.assert_array_equal(out_uns["mean"], out_fallback["mean"])
    np.testing.assert_array_equal(out_uns["std"], out_fallback["std"])
    print("PASS  uns path matches on-the-fly when stats identical")


def main():
    test_klein_path_pca()
    test_klein_path_dm()
    test_cordblood_path_pca()
    test_cordblood_path_pca_harmony()
    test_cordblood_path_dm()
    test_non_scaled_key_returns_none()
    test_missing_raw_key_raises()
    test_uns_path_does_not_need_raw_obsm()
    test_uns_scaler_smaller_than_n_dims_raises()
    test_uns_path_matches_on_the_fly_when_stats_match()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
