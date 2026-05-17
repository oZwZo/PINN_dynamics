#!/usr/bin/env python
"""
Evaluate every checkpoint produced by the C1 λ_growth sweep against the GT
spline fields stored in biomodal_g_norm_D_params.ckpt.

The GT is `MultiDim_CubicSpline` — separable per dimension:
  v(s) = [ v_d(s_d) ]                   (vector)
  g(s) = sum_d g_d(s_d)                 (scalar, collapse=True)
  D(s) = [ D_d(s_d) ] (or sum if collapsed; we read both forms and report both)

For each (arm, seed):
  - locate the lowest-val_loss .ckpt under logs/.../lightning_logs/version_*/checkpoints
  - load `pde_params.load_from_checkpoint(...)`
  - run model.predict_param(train_DS) at the observed cells (timepoints 0..4)
  - build GT fields at the same cells via scipy.interpolate.CubicSpline
  - compute relative L² and Pearson r for each of g, v, D
Outputs:
  logs/synthetic_FP_5D_ablation/eval/per_seed.csv
  logs/synthetic_FP_5D_ablation/eval/summary.csv  (mean ± SD across seeds)
  logs/synthetic_FP_5D_ablation/eval/sensitivity_curve.png
"""
import argparse
import glob
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.interpolate import CubicSpline
from scipy.stats import pearsonr

import pseudodynamics as pdp                         # noqa: F401  (ensures package import paths)
from pseudodynamics import models, reader


REPO_ROOT = Path("/rds/user/wz369/hpc-work/pseudodynamics_plus")
ABL_ROOT_v2 = REPO_ROOT / "logs/synthetic_FP_5D_ablation"     # main_train.py path
ABL_ROOT_v3 = REPO_ROOT / "logs/synthetic_FP_5D_synds"        # c1_train_synds.py path
GT_CKPT = REPO_ROOT / "data/synthesized_data/5Dim_ncs_synparam_Jan23/biomodal_g_norm_D_params.ckpt"
H5AD = REPO_ROOT / "data/synthetic_FP_5D.h5ad"


def build_gt_fns(gt_ckpt_path: Path):
    """
    Returns (g_fn, v_fn, D_fn) each callable on (n_cells, 5) → respectively
    (n_cells,), (n_cells, 5), (n_cells, 5).
    Per-dim cubic splines per MultiDim_CubicSpline.
    """
    try:
        obj = torch.load(gt_ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(
            f"failed to load GT ckpt at {gt_ckpt_path}: {type(e).__name__}: {e}. "
            f"Check that the file exists and was pickled with a compatible torch version."
        ) from e
    for key in ("x", "g", "v", "D"):
        if key not in obj:
            raise KeyError(f"GT ckpt missing key '{key}'; have {list(obj)}")
    x = obj["x"].numpy()       # (5, 12) knot positions
    g_knots = obj["g"].numpy() # (5, 12)
    v_knots = obj["v"].numpy()
    D_knots = obj["D"].numpy()
    assert x.shape == (5, 12) and g_knots.shape == (5, 12), \
        f"unexpected GT shapes: x={x.shape} g={g_knots.shape}"

    g_splines = [CubicSpline(x[d], g_knots[d]) for d in range(5)]
    v_splines = [CubicSpline(x[d], v_knots[d]) for d in range(5)]
    D_splines = [CubicSpline(x[d], D_knots[d]) for d in range(5)]

    def g_fn(s):     # (n, 5) → (n,) sum-collapse
        return np.stack([g_splines[d](s[:, d]) for d in range(5)], axis=-1).sum(axis=-1)

    def v_fn(s):     # (n, 5) → (n, 5)
        return np.stack([v_splines[d](s[:, d]) for d in range(5)], axis=-1)

    def D_fn(s):     # (n, 5) → (n, 5)
        return np.stack([D_splines[d](s[:, d]) for d in range(5)], axis=-1)

    return g_fn, v_fn, D_fn


def metrics(pred: np.ndarray, true: np.ndarray, name: str):
    """Relative-L² and Pearson r. Operates on flattened arrays."""
    pred_f = np.asarray(pred).reshape(-1)
    true_f = np.asarray(true).reshape(-1)
    rel_l2 = np.linalg.norm(pred_f - true_f) / max(np.linalg.norm(true_f), 1e-8)
    pearson = pearsonr(pred_f, true_f).statistic if pred_f.std() > 0 else 0.0
    return {f"{name}_rel_l2": float(rel_l2), f"{name}_pearson": float(pearson)}


def find_best_ckpt(arm_seed_dir: Path) -> Optional[Path]:
    """Pick the .ckpt with the lowest val_loss in the filename.

    Searches recursively to handle both layouts:
      - v2 (main_train.py):     pde_params_tsense/lightning_logs/version_*/checkpoints/*.ckpt
      - v3 (c1_train_synds.py): lightning_logs/checkpoints/best_val-*.ckpt
    For v3, prefer `best_val-*` over `best_tot-*` since c1_eval.py monitors val_loss.
    """
    cands = list(arm_seed_dir.glob("**/checkpoints/*.ckpt"))
    if not cands:
        return None

    # v3 may include `best_tot-*` ckpts; drop them so we only rank val-monitored ckpts.
    val_cands = [c for c in cands if not c.stem.startswith("best_tot-")]
    if val_cands:
        cands = val_cands

    def loss_of(p: Path) -> float:
        try:
            tag = p.stem.split("val_loss=")[-1]
            return float(tag)
        except Exception:
            return float("inf")

    return min(cands, key=loss_of)


def eval_one(ckpt_path: Path, adata, train_DS, gt_fns):
    g_fn, v_fn, D_fn = gt_fns
    model = models.pde_params.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    model.eval()
    g_hat, v_hat, D_hat = model.predict_param(train_DS)

    # predict_param iterates train_DS.s in chunks without shuffling, so the
    # returned tensors are in the same order as train_DS.s.
    # Shapes from predict_param:
    #   g_hat: (n_timepoint, n_cells_per_t)            scalar per cell
    #   v_hat: (n_timepoint, n_cells_per_t, v_dim)     vector per cell
    #   D_hat: (n_timepoint, n_cells_per_t)            if collapse_D=True (default)
    #          (n_timepoint, n_cells_per_t, 5)         if collapse_D=False
    g_hat = np.asarray(g_hat)
    v_hat = np.asarray(v_hat)
    D_hat = np.asarray(D_hat)
    assert g_hat.ndim == 2,                            f"expected g_hat ndim=2, got {g_hat.shape}"
    assert v_hat.ndim == 3 and v_hat.shape[-1] == 5,   f"expected v_hat (.,.,5), got {v_hat.shape}"
    assert D_hat.ndim in (2, 3),                       f"expected D_hat ndim in (2,3), got {D_hat.shape}"

    g_hat = g_hat.reshape(-1)                                          # (n_total,)
    v_hat = v_hat.reshape(-1, v_hat.shape[-1])                         # (n_total, 5)
    if D_hat.ndim == 2:
        D_hat_flat = D_hat.reshape(-1)                                 # (n_total,) — collapsed scalar
        D_collapsed = True
    else:
        D_hat_flat = D_hat.reshape(-1, D_hat.shape[-1])                # (n_total, 5)
        D_collapsed = False

    # GT at the same observed (s, t). Order matches train_DS.s by construction
    # of predict_param (sequential chunks, no shuffle).
    s = np.asarray(train_DS.s)                                          # (n_total, 5)
    g_true = g_fn(s)
    v_true = v_fn(s)
    D_true = D_fn(s)
    D_true_compare = D_true.sum(axis=-1) if D_collapsed else D_true
    D_hat = D_hat_flat

    out = {}
    out.update(metrics(g_hat, g_true, "g"))
    out.update(metrics(v_hat, v_true, "v"))
    out.update(metrics(D_hat, D_true_compare, "D"))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=None,
                        help="Default writes per-version eval into <root>/eval. Pass to override.")
    parser.add_argument("--version", choices=["v2", "v3", "both"], default="both",
                        help="Which sweep root(s) to evaluate.")
    args = parser.parse_args()

    roots = []
    if args.version in ("v2", "both") and ABL_ROOT_v2.exists():
        roots.append(("v2", ABL_ROOT_v2))
    if args.version in ("v3", "both") and ABL_ROOT_v3.exists():
        roots.append(("v3", ABL_ROOT_v3))
    if not roots:
        print("no sweep roots found"); return

    # Build the dataset once (shared across all 15 evals)
    adata = sc.read_h5ad(H5AD)
    ds_kws = dict(
        timepoint_idx=list(range(11)),
        n_dimension=5,
        cellstate_key="X_data",
        knn_volume=False,
        log_transform=False,
        norm_time="min_minus",
        deltax_key="Delta_DM",
        kde_kws={"bw_method": None},
        batchsize=200,
    )
    # split=None: use the full 22000 cells. predict_param requires
    # train_DS.s and train_DS.t_b to have the same length, but
    # reader.subset_dataset() shrinks `s` without shrinking `t_b`. Since g, v, D
    # are learned functions evaluable anywhere in (s, t), eval'ing on all cells
    # (including the held-out 10% val + 10% test) is fine and gives more samples.
    train_DS = reader.TwoTimpepoint_AnnDS(AnnData=adata, split=None, **ds_kws)
    gt_fns = build_gt_fns(GT_CKPT)

    # Iterate over every (version, arm, seed) directory
    rows = []
    for version, root in roots:
        for arm_dir in sorted(root.glob("lambdag_*")):
            for seed_dir in sorted(arm_dir.glob("seed_*")):
                ckpt = find_best_ckpt(seed_dir)
                if ckpt is None:
                    print(f"[skip] no ckpt in {seed_dir}")
                    continue
                try:
                    r = eval_one(ckpt, adata, train_DS, gt_fns)
                except Exception as e:
                    print(f"[error] {seed_dir}: {type(e).__name__}: {e}")
                    continue
                r["version"] = version
                r["arm"] = arm_dir.name
                r["seed"] = int(seed_dir.name.split("_")[-1])
                r["ckpt"] = str(ckpt.relative_to(REPO_ROOT))
                r["lambda_growth"] = float(arm_dir.name.split("lambdag_")[-1].replace("p", "."))
                rows.append(r)
                print(f"[ok] {version} {arm_dir.name} seed={r['seed']}  "
                      f"g_l2={r['g_rel_l2']:.3f}  v_l2={r['v_rel_l2']:.3f}  D_l2={r['D_rel_l2']:.3f}")

    if not rows:
        print("no rows; exiting")
        return

    df = pd.DataFrame(rows)
    if args.out_dir is None:
        # Write to each version's own eval/ for clean separation
        for version, root in roots:
            sub = df[df["version"] == version]
            if len(sub) == 0:
                continue
            ed = root / "eval"
            ed.mkdir(parents=True, exist_ok=True)
            sub.to_csv(ed / "per_seed.csv", index=False)
            print(f"wrote {ed}/per_seed.csv ({len(sub)} rows)")
        out_dir = roots[-1][1] / "eval"   # use the last version's dir for combined plots/summary
    else:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "per_seed.csv", index=False)

    summary = (
        df.groupby(["lambda_growth", "arm"])[
            ["g_rel_l2", "v_rel_l2", "D_rel_l2", "g_pearson", "v_pearson", "D_pearson"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv(out_dir / "summary.csv")

    # sensitivity-curve plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
    for ax, field in zip(axes, ["g", "v", "D"]):
        col = f"{field}_rel_l2"
        agg = df.groupby("lambda_growth")[col].agg(["mean", "std"]).reset_index()
        x = agg["lambda_growth"].copy().replace(0, 1e-3)   # log-axis: 0 → tiny ε
        ax.errorbar(x, agg["mean"], yerr=agg["std"], marker="o", capsize=3)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\lambda_{\mathrm{growth}}$  (0 shown as $10^{-3}$)")
        ax.set_ylabel(f"relative $L^2$ error: {field}")
        ax.set_title(field)
    plt.tight_layout()
    plt.savefig(out_dir / "sensitivity_curve.png", dpi=150)
    plt.close()

    print(f"wrote {out_dir}/per_seed.csv, summary.csv, sensitivity_curve.png")


if __name__ == "__main__":
    main()
