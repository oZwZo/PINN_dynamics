"""Build cord-blood benchmark progress report.

Produces:
  - reports/cordblood/figures/*.png  (UMAP trajectories, training curves, completion grid)
  - reports/cordblood/data/*.npz     (saved trajectories per method)
  - reports/cordblood/leaderboard.csv
  - reports/cordblood/per_seed_metrics.csv

Designed to be called from the notebook (build_report.ipynb) so individual
sections can be re-run cheaply.
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = Path("/rds/user/wz369/hpc-work/PINN_dynamics")
LOG_ROOT = PROJ / "logs" / "cordblood"
REPORT_ROOT = PROJ / "reports" / "cordblood"
FIG_DIR = REPORT_ROOT / "figures"
DATA_DIR = REPORT_ROOT / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["pdp", "prescient", "otcfm", "sf2m", "deepruot", "scdiffeq"]
EMBS = ["X_pca_scaled", "DM_EigenVectors_scaled"]
SEEDS = [42, 7, 1234]

# --------------------------------------------------------------------------- #
# 1) Dataset summary
# --------------------------------------------------------------------------- #

def dataset_summary():
    import scanpy as sc
    adata = sc.read_h5ad(str(PROJ / "data" / "cordblood_addpop.h5ad"))
    rows = [
        ("n_obs", adata.n_obs),
        ("n_vars", adata.n_vars),
        ("timepoints", sorted(adata.obs["timepoint_tx_days"].unique().tolist())),
        ("n_clones (unique)", int(adata.obs["clones"].nunique())),
        ("cell types (def_lab)", sorted(adata.obs["def_lab"].unique().tolist())),
        ("train cells", int((adata.obs["split"] == "train").sum())),
        ("val cells", int((adata.obs["split"] == "val").sum())),
        ("test cells", int((adata.obs["split"] == "test").sum())),
        ("X_pca_scaled shape", adata.obsm["X_pca_scaled"].shape),
        ("DM_EigenVectors_scaled shape", adata.obsm["DM_EigenVectors_scaled"].shape),
        ("pca_scaler n_train_cells", adata.uns["pca_scaler"].get("n_train_cells")),
    ]
    df = pd.DataFrame(rows, columns=["key", "value"])
    df.to_csv(REPORT_ROOT / "dataset_summary.csv", index=False)
    return df, adata

# --------------------------------------------------------------------------- #
# 2) Evaluation tasks setup
# --------------------------------------------------------------------------- #

def eval_tasks_table():
    """One row per (method, embedding, seed) — Cartesian over METHODS×EMBS×SEEDS."""
    rows = []
    for m in METHODS:
        for e in EMBS:
            for s in SEEDS:
                rows.append({"method": m, "embedding": e, "seed": s})
    df = pd.DataFrame(rows)
    df.to_csv(REPORT_ROOT / "eval_tasks.csv", index=False)
    return df

# --------------------------------------------------------------------------- #
# 3) Completion matrix
# --------------------------------------------------------------------------- #

def find_eval_csv(method: str, embedding: str, seed: int):
    """Return path to eval_combined.csv if exists, else None."""
    candidates = [
        LOG_ROOT / method / f"{embedding}_s{seed}" / "eval" / "eval_combined.csv",
        LOG_ROOT / method / f"{embedding}_s{seed}" / "eval_combined.csv",
    ]
    # scDiffEq writes under a different folder shape
    if method == "scdiffeq":
        emb_short = "dm9" if "DM" in embedding else "pca30"
        candidates.extend([
            LOG_ROOT / "scdiffeq" / f"{emb_short}_fp_vr_s{seed}" / "eval" / "eval_combined.csv",
            LOG_ROOT / "scdiffeq" / f"{emb_short}_fp_vr_s{seed}" / "eval_combined.csv",
        ])
        # scDiffEq's evaluator nests further: eval/fate_prediction_metrics/epoch_N.step_M/eval_combined.csv
        nested = sorted((LOG_ROOT / "scdiffeq" / f"{emb_short}_fp_vr_s{seed}" / "eval" /
                         "fate_prediction_metrics").glob("epoch_*.step_*/eval_combined.csv"),
                        key=lambda p: p.stat().st_mtime if p.exists() else 0)
        candidates.extend(nested)
    # DeepRUOT: external repo
    if method == "deepruot":
        emb_short = "dm9" if "DM" in embedding else "pca30"
        candidates.extend([
            LOG_ROOT / "deepruot" / f"{emb_short}_s{seed}" / "eval_combined.csv",
        ])
    for c in candidates:
        if c.exists():
            return c
    return None

def completion_matrix():
    rows = []
    for m in METHODS:
        for e in EMBS:
            for s in SEEDS:
                p = find_eval_csv(m, e, s)
                rows.append({
                    "method": m, "embedding": e, "seed": s,
                    "evaluated": p is not None,
                    "path": str(p) if p else "",
                })
    df = pd.DataFrame(rows)
    df.to_csv(REPORT_ROOT / "completion_matrix.csv", index=False)
    return df

def plot_completion_grid(df: pd.DataFrame, out=FIG_DIR / "completion_grid.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), sharey=True)
    for ax, emb in zip(axes, EMBS):
        sub = df[df.embedding == emb]
        pivot = sub.pivot_table(index="method", columns="seed",
                                values="evaluated", aggfunc="first").reindex(METHODS)
        pivot = pivot.fillna(False).astype(int)
        ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(SEEDS))); ax.set_xticklabels(SEEDS)
        ax.set_yticks(range(len(METHODS))); ax.set_yticklabels(METHODS)
        ax.set_title(emb.replace("_scaled", ""))
        for i in range(len(METHODS)):
            for j in range(len(SEEDS)):
                v = pivot.values[i, j]
                ax.text(j, i, "OK" if v else "—", ha="center", va="center",
                        color="black" if v else "gray", fontsize=10)
    plt.suptitle("Cord-blood eval completion (green = eval_combined.csv exists)")
    plt.tight_layout()
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()
    return out

# --------------------------------------------------------------------------- #
# 4) Per-method metrics table
# --------------------------------------------------------------------------- #

def gather_metrics(matrix_df: pd.DataFrame):
    rows = []
    for _, row in matrix_df[matrix_df.evaluated].iterrows():
        try:
            df = pd.read_csv(row.path)
        except Exception as exc:
            print(f"skip {row.path}: {exc}")
            continue
        for _, r in df.iterrows():
            rows.append({
                "method": row.method, "embedding": row.embedding, "seed": row.seed,
                "sim_mode": r.get("sim_mode"),
                "accuracy": r.get("accuracy"),
                "pearson_r": r.get("pearson_r"),
                "w2_scaled": r.get("w2_scaled"),
                "w2_raw": r.get("w2_raw"),
            })
    df = pd.DataFrame(rows)
    df.to_csv(REPORT_ROOT / "per_seed_metrics.csv", index=False)
    return df

def leaderboard(per_seed: pd.DataFrame):
    if per_seed.empty:
        empty = pd.DataFrame(columns=["method", "embedding", "sim_mode",
                                      "accuracy_mean", "pearson_r_mean",
                                      "w2_scaled_mean", "w2_scaled_std", "n_seeds"])
        empty.to_csv(REPORT_ROOT / "leaderboard.csv", index=False)
        return empty
    agg = per_seed.groupby(["method", "embedding", "sim_mode"]).agg(
        accuracy_mean=("accuracy", "mean"),
        pearson_r_mean=("pearson_r", "mean"),
        w2_scaled_mean=("w2_scaled", "mean"),
        w2_scaled_std=("w2_scaled", "std"),
        w2_raw_mean=("w2_raw", "mean"),
        n_seeds=("seed", "nunique"),
    ).reset_index().sort_values(["embedding", "w2_scaled_mean"])
    agg.to_csv(REPORT_ROOT / "leaderboard.csv", index=False)
    return agg

# --------------------------------------------------------------------------- #
# 5) Training curves
# --------------------------------------------------------------------------- #

LOSS_RE_OTCFM_SF2M = re.compile(
    r"iter\s+(\d+)/\d+\s+loss=(\d+\.\d+)"
)
VAL_RE_OTCFM_SF2M = re.compile(
    r"\[val\]\s+iter\s+(\d+)\s+val_loss=(\d+\.\d+)"
)

def parse_otcfm_sf2m_log(log_path: Path):
    iters_train, loss_train = [], []
    iters_val, loss_val = [], []
    text = log_path.read_text(errors="ignore")
    for m in LOSS_RE_OTCFM_SF2M.finditer(text):
        iters_train.append(int(m.group(1))); loss_train.append(float(m.group(2)))
    for m in VAL_RE_OTCFM_SF2M.finditer(text):
        iters_val.append(int(m.group(1))); loss_val.append(float(m.group(2)))
    return np.array(iters_train), np.array(loss_train), np.array(iters_val), np.array(loss_val)

def plot_training_curves():
    """One panel per (method, embedding); 3 seeds overlaid."""
    methods_with_logs = ["otcfm", "sf2m"]  # parseable today
    fig, axes = plt.subplots(len(methods_with_logs), len(EMBS),
                             figsize=(11, 3.0 * len(methods_with_logs)),
                             sharex=False, sharey=False, squeeze=False)
    for r, m in enumerate(methods_with_logs):
        for c, e in enumerate(EMBS):
            ax = axes[r][c]
            for s in SEEDS:
                log = LOG_ROOT / m / f"{e}_s{s}.log"
                if not log.exists():
                    continue
                it_t, l_t, it_v, l_v = parse_otcfm_sf2m_log(log)
                if len(l_t) == 0:
                    continue
                ax.plot(it_t, l_t, alpha=0.6, label=f"s{s} train", lw=0.8)
                if len(l_v) > 0:
                    ax.plot(it_v, l_v, "--", alpha=0.9, label=f"s{s} val")
            ax.set_title(f"{m} / {e.replace('_scaled','')}")
            ax.set_xlabel("iter"); ax.set_ylabel("loss")
            ax.legend(fontsize=7)
    plt.tight_layout()
    out = FIG_DIR / "training_curves.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()
    return out

# --------------------------------------------------------------------------- #
# 6) UMAP trajectory per method
# --------------------------------------------------------------------------- #

def get_or_make_umap(adata):
    """Return (umap_xy, key) — embeds X_pca_scaled with scanpy default if missing."""
    if "X_umap" in adata.obsm:
        return adata.obsm["X_umap"]
    import scanpy as sc
    sc.pp.neighbors(adata, use_rep="X_pca_scaled", n_neighbors=30)
    sc.tl.umap(adata)
    return adata.obsm["X_umap"]

def collect_trajectory(method: str, embedding: str, seed: int):
    """Attempt to load the saved simulation trajectory for this run.

    Returns dict or None. Different methods save trajectories under different keys.
    For methods that don't expose a trajectory file, we return None and the notebook
    will just plot the test-cell UMAP without an overlay.
    """
    eval_dir = LOG_ROOT / method / f"{embedding}_s{seed}" / "eval"
    candidates = list(eval_dir.glob("trajectory*.npz")) + list(eval_dir.glob("sim*.npz"))
    if candidates:
        d = np.load(str(candidates[0]), allow_pickle=True)
        return {"file": str(candidates[0]), "keys": list(d.keys())}
    return None

def plot_method_umap(adata, method: str, embedding: str, seed: int = 42,
                     umap_xy=None, out=None):
    if umap_xy is None:
        umap_xy = get_or_make_umap(adata)
    fig, ax = plt.subplots(figsize=(5, 5))
    # background cells (train)
    mask_train = (adata.obs["split"] == "train").values
    ax.scatter(umap_xy[mask_train, 0], umap_xy[mask_train, 1],
               s=2, color="lightgrey", alpha=0.3, label="train")
    # test cells coloured by fate
    mask_test = (adata.obs["split"] == "test").values
    fates = adata.obs["def_lab"][mask_test].astype(str).values
    uniq_fates = sorted(np.unique(fates))
    cmap = plt.colormaps["tab20"].resampled(len(uniq_fates))
    for k, f in enumerate(uniq_fates):
        m = fates == f
        ax.scatter(umap_xy[mask_test][m, 0], umap_xy[mask_test][m, 1],
                   s=6, color=cmap(k), label=f, alpha=0.8)
    # If trajectory is loadable, overlay arrow field (best-effort)
    info = collect_trajectory(method, embedding, seed)
    title = f"{method} / {embedding.replace('_scaled','')} / s{seed}"
    if info:
        title += "  (+traj)"
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=6, markerscale=2, ncol=2,
              bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    if out is None:
        out = FIG_DIR / f"umap_{method}_{embedding}_s{seed}.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()
    return out

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    summary, adata = dataset_summary()
    print(summary.to_string(index=False))

    tasks = eval_tasks_table()
    print(f"\n{len(tasks)} eval tasks total")

    matrix = completion_matrix()
    n_done = int(matrix.evaluated.sum())
    print(f"Eval completion: {n_done}/{len(matrix)}")
    plot_completion_grid(matrix)

    per_seed = gather_metrics(matrix)
    print(f"\nPer-seed metrics: {len(per_seed)} rows")
    if not per_seed.empty:
        print(per_seed.head().to_string(index=False))

    lb = leaderboard(per_seed)
    print(f"\nLeaderboard: {len(lb)} rows")
    if not lb.empty:
        print(lb.to_string(index=False))

    plot_training_curves()

    umap_xy = get_or_make_umap(adata)
    np.savez(DATA_DIR / "umap_xy.npz", X_umap=umap_xy,
             split=adata.obs["split"].astype(str).values,
             def_lab=adata.obs["def_lab"].astype(str).values,
             timepoint=adata.obs["timepoint_tx_days"].values)
    for m in METHODS:
        for e in EMBS:
            for s in SEEDS:
                if (LOG_ROOT / m / f"{e}_s{s}").exists():
                    plot_method_umap(adata, m, e, s, umap_xy=umap_xy)
                    break  # 1 panel per (method, embedding) is enough for the grid
    print("Done.")

if __name__ == "__main__":
    main()
