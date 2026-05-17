#!/usr/bin/env python3
"""
Evaluate pdp+ D-field recovery on the synthetic Fokker-Planck dataset.

Usage (single checkpoint):
    python 08_eval_synthetic_FP.py \
        --config_path logs/synthetic_FP_ablation/baseline/seed_0/pde_params_tsense/V1_config.json

Usage (sweep all arms × seeds):
    python 08_eval_synthetic_FP.py \
        --sweep_root logs/synthetic_FP_ablation \
        --arms baseline lambdaD_0 lambdaD_0.01 lambdaD_10

Outputs per checkpoint:
    <output_dir>/D_recovery.csv        — per-timepoint metrics
    <output_dir>/D_recovery_plots.pdf  — parity + spatial comparison panels
"""

import argparse
import os
import pickle
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_ground_truth(gt_path):
    with open(gt_path, "rb") as f:
        gt = pickle.load(f)
    return gt["grid_x"], gt["grid_y"], gt["D_grid"], gt["v_grid"], gt["g_grid"]


def find_best_ckpt(ckpt_dir):
    """Return path to checkpoint with the highest val_loss (least negative for NLL)."""
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if not ckpts:
        return None
    losses = []
    for c in ckpts:
        m = re.search(r"val_loss=(-?[\d.]+)\.ckpt$", c)
        if m:
            losses.append(float(m.group(1)))
        else:
            losses.append(float("-inf"))
    return os.path.join(ckpt_dir, ckpts[int(np.argmax(losses))])


def predict_D_on_grid(model, grid_x, grid_y, timepoints, device="cpu"):
    """Evaluate model.D(s, t) on the ground-truth grid at each timepoint.

    Returns: dict  t -> D_pred array (same shape as grid_x)
    """
    nx, ny = grid_x.shape
    s_flat = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)  # (N, 2)
    s_ts = torch.tensor(s_flat, dtype=torch.float32, device=device)

    D_by_t = {}
    with torch.no_grad():
        for t_val in timepoints:
            t_ts = torch.full((s_ts.shape[0], 1), t_val, dtype=torch.float32, device=device)
            D_pred = model.D(s_ts, t_ts)
            D_by_t[t_val] = D_pred.cpu().numpy().reshape(nx, ny)
    return D_by_t


def compute_metrics(D_true, D_pred):
    """Compute recovery metrics between two 2-D fields (flattened)."""
    dt = D_true.ravel()
    dp = D_pred.ravel()
    pearson_r, pearson_p = stats.pearsonr(dt, dp)
    spearman_r, spearman_p = stats.spearmanr(dt, dp)
    slope, intercept, _, _, _ = stats.linregress(dt, dp)
    rmse = np.sqrt(np.mean((dt - dp) ** 2))
    mae = np.mean(np.abs(dt - dp))
    r2 = 1 - np.sum((dt - dp) ** 2) / np.sum((dt - dt.mean()) ** 2)
    return {
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "slope": slope,
        "intercept": intercept,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def compute_diffusion_flux(D_field, grid_x, grid_y, u_field):
    """Compute diffusion flux  div(D * grad(u))  on a regular grid via finite differences.

    This measures the actual contribution of diffusion to density transport.
    Uses central differences; boundary points are excluded from the returned flux.
    """
    dx = grid_x[0, 1] - grid_x[0, 0]
    dy = grid_y[1, 0] - grid_y[0, 0]

    # grad(u)
    du_dx = np.gradient(u_field, dx, axis=1)
    du_dy = np.gradient(u_field, dy, axis=0)

    # D * grad(u)
    Jx = D_field * du_dx
    Jy = D_field * du_dy

    # div(J) = dJx/dx + dJy/dy
    div_J = np.gradient(Jx, dx, axis=1) + np.gradient(Jy, dy, axis=0)
    return div_J


def make_true_density_on_grid(adata, grid_x, grid_y, timepoints, bw=0.05):
    """KDE-estimate the true cell density on the ground-truth grid at each timepoint."""
    from scipy.stats import gaussian_kde

    u_by_t = {}
    for t_val in timepoints:
        mask = adata.obs["timepoint"] == t_val
        xy = adata[mask].obsm["cellstate"]
        kde = gaussian_kde(xy.T, bw_method=bw)
        positions = np.vstack([grid_x.ravel(), grid_y.ravel()])
        u_by_t[t_val] = kde(positions).reshape(grid_x.shape)
    return u_by_t


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_D_recovery(grid_x, grid_y, D_true, D_by_t, metrics_by_t, pdf_path):
    """One page per timepoint: parity plot + spatial maps (true, pred, diff)."""
    timepoints = sorted(D_by_t.keys())
    with PdfPages(pdf_path) as pdf:
        for t_val in timepoints:
            D_pred = D_by_t[t_val]
            m = metrics_by_t[t_val]

            fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
            fig.suptitle(
                f"t = {t_val:.1f}   |   r = {m['pearson_r']:.3f}   "
                f"slope = {m['slope']:.3f}   RMSE = {m['rmse']:.4f}   R² = {m['r2']:.3f}",
                fontsize=12,
            )

            # parity
            ax = axes[0]
            ax.scatter(D_true.ravel(), D_pred.ravel(), s=1, alpha=0.3)
            lims = [min(D_true.min(), D_pred.min()), max(D_true.max(), D_pred.max())]
            ax.plot(lims, lims, "k--", lw=0.8)
            ax.set_xlabel("D true")
            ax.set_ylabel("D pred")
            ax.set_title("Parity")
            ax.set_aspect("equal")

            # true D
            vmin, vmax = D_true.min(), D_true.max()
            ax = axes[1]
            im = ax.pcolormesh(grid_x, grid_y, D_true, vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set_title("D true")
            ax.set_aspect("equal")
            plt.colorbar(im, ax=ax, fraction=0.046)

            # pred D
            ax = axes[2]
            im = ax.pcolormesh(grid_x, grid_y, D_pred, vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set_title("D pred")
            ax.set_aspect("equal")
            plt.colorbar(im, ax=ax, fraction=0.046)

            # difference
            ax = axes[3]
            diff = D_pred - D_true
            vlim = max(abs(diff.min()), abs(diff.max()))
            im = ax.pcolormesh(grid_x, grid_y, diff, vmin=-vlim, vmax=vlim, cmap="RdBu_r")
            ax.set_title("D pred − D true")
            ax.set_aspect("equal")
            plt.colorbar(im, ax=ax, fraction=0.046)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # -- Flux comparison page (if density available) --
        # This is appended after the per-timepoint pages


def plot_flux_comparison(grid_x, grid_y, flux_true_by_t, flux_pred_by_t, pdf_path):
    """Append flux comparison pages to an existing PDF."""
    timepoints = sorted(flux_true_by_t.keys())
    with PdfPages(pdf_path) as pdf:
        for t_val in timepoints:
            ft = flux_true_by_t[t_val]
            fp = flux_pred_by_t[t_val]

            flux_corr, _ = stats.pearsonr(ft.ravel(), fp.ravel())

            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
            fig.suptitle(f"Diffusion flux  div(D·∇u)  at t={t_val:.1f}   |   flux Pearson r = {flux_corr:.3f}", fontsize=12)

            vlim = max(abs(ft).max(), abs(fp).max())
            for ax, data, title in zip(axes[:2], [ft, fp], ["True D flux", "Pred D flux"]):
                im = ax.pcolormesh(grid_x, grid_y, data, vmin=-vlim, vmax=vlim, cmap="RdBu_r")
                ax.set_title(title)
                ax.set_aspect("equal")
                plt.colorbar(im, ax=ax, fraction=0.046)

            ax = axes[2]
            ax.scatter(ft.ravel(), fp.ravel(), s=1, alpha=0.3)
            lims = [-vlim, vlim]
            ax.plot(lims, lims, "k--", lw=0.8)
            ax.set_xlabel("True flux")
            ax.set_ylabel("Pred flux")
            ax.set_title("Flux parity")
            ax.set_aspect("equal")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


# ---------------------------------------------------------------------------
# Single-checkpoint evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoint(config_path, gt_path, data_path, output_dir, device="cpu"):
    """Run full D-recovery evaluation for one checkpoint. Returns metrics dict."""
    import pseudodynamics as pdp

    # load config & model
    config = pdp.ExperimentConfig(config_path)
    ckpt_dir = os.path.join(config.experiment_config["checkpoint_dir"], "checkpoints")
    ckpt_path = find_best_ckpt(ckpt_dir)
    print(f"  Loading checkpoint: {os.path.basename(ckpt_path)}")

    pde_model = pdp.models.pde_params.load_from_checkpoint(ckpt_path).to(device)
    pde_model.eval()

    # load ground truth
    grid_x, grid_y, D_true, v_true, g_true = load_ground_truth(gt_path)
    timepoints = [0.0, 1.0, 2.0, 3.0, 4.0]

    # predict D on grid
    D_by_t = predict_D_on_grid(pde_model, grid_x, grid_y, timepoints, device=device)

    # compute per-timepoint metrics
    metrics_by_t = {}
    for t_val in timepoints:
        metrics_by_t[t_val] = compute_metrics(D_true, D_by_t[t_val])

    # save metrics CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "D_recovery.csv")
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timepoint"] + list(next(iter(metrics_by_t.values())).keys()))
        writer.writeheader()
        for t_val in timepoints:
            row = {"timepoint": t_val, **metrics_by_t[t_val]}
            writer.writerow(row)
    print(f"  Metrics saved: {csv_path}")

    # D-field plots
    pdf_path = os.path.join(output_dir, "D_recovery_plots.pdf")
    plot_D_recovery(grid_x, grid_y, D_true, D_by_t, metrics_by_t, pdf_path)
    print(f"  Plots saved: {pdf_path}")

    # diffusion flux comparison (using true density from data)
    import scanpy as sc
    adata = sc.read_h5ad(data_path)
    u_by_t = make_true_density_on_grid(adata, grid_x, grid_y, timepoints)

    flux_true_by_t = {}
    flux_pred_by_t = {}
    flux_metrics_by_t = {}
    for t_val in timepoints:
        flux_true_by_t[t_val] = compute_diffusion_flux(D_true, grid_x, grid_y, u_by_t[t_val])
        flux_pred_by_t[t_val] = compute_diffusion_flux(D_by_t[t_val], grid_x, grid_y, u_by_t[t_val])
        fc, _ = stats.pearsonr(flux_true_by_t[t_val].ravel(), flux_pred_by_t[t_val].ravel())
        flux_metrics_by_t[t_val] = fc

    flux_pdf = os.path.join(output_dir, "D_flux_comparison.pdf")
    plot_flux_comparison(grid_x, grid_y, flux_true_by_t, flux_pred_by_t, flux_pdf)
    print(f"  Flux plots saved: {flux_pdf}")

    # append flux metrics to CSV
    flux_csv = os.path.join(output_dir, "D_flux_metrics.csv")
    with open(flux_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timepoint", "flux_pearson_r"])
        writer.writeheader()
        for t_val in timepoints:
            writer.writerow({"timepoint": t_val, "flux_pearson_r": flux_metrics_by_t[t_val]})
    print(f"  Flux metrics saved: {flux_csv}")

    return metrics_by_t, flux_metrics_by_t


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # single-checkpoint mode
    p.add_argument("--config_path", type=str, help="Path to V*_config.json for a single run")

    # sweep mode
    p.add_argument("--sweep_root", type=str, help="Root dir containing arm subdirs (e.g. logs/synthetic_FP_ablation)")
    p.add_argument("--arms", nargs="+", default=["baseline", "lambdaD_0", "lambdaD_0.01", "lambdaD_10"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--version", type=int, default=None, help="Lightning version to use (default: latest)")

    # common
    p.add_argument("--gt_path", type=str, default="data/synthetic_FP_ground_truth.pkl")
    p.add_argument("--data_path", type=str, default="data/synthetic_FP.h5ad")
    p.add_argument("--output_dir", type=str, default="results/synthetic_FP_eval")
    p.add_argument("--device", type=str, default="cpu")

    return p.parse_args()


def resolve_config(sweep_root, arm, seed, version=None):
    """Find the config JSON for a given arm/seed, preferring the latest version."""
    base = Path(sweep_root) / arm / f"seed_{seed}" / "pde_params_tsense"
    configs = sorted(base.glob("V*_config.json"), reverse=True)
    if version is not None:
        target = base / f"V{version}_config.json"
        return str(target) if target.exists() else None
    return str(configs[0]) if configs else None


def main():
    args = parse_args()

    if args.config_path:
        # single-checkpoint mode
        print(f"Evaluating: {args.config_path}")
        evaluate_checkpoint(args.config_path, args.gt_path, args.data_path, args.output_dir, args.device)

    elif args.sweep_root:
        # sweep mode
        import csv
        summary_rows = []
        for arm in args.arms:
            for seed in args.seeds:
                config_path = resolve_config(args.sweep_root, arm, seed, args.version)
                if config_path is None or not os.path.exists(config_path):
                    print(f"SKIP {arm}/seed_{seed}: no config found")
                    continue

                # check if checkpoint exists
                import json
                with open(config_path) as f:
                    cfg = json.load(f)
                ckpt_dir = cfg["experiment_config"]["checkpoint_dir"]
                ckpt_dir = os.path.join(ckpt_dir, "checkpoints")
                if not os.path.isdir(ckpt_dir) or not any(f.endswith(".ckpt") for f in os.listdir(ckpt_dir)):
                    print(f"SKIP {arm}/seed_{seed}: no checkpoints in {ckpt_dir}")
                    continue

                out = os.path.join(args.output_dir, arm, f"seed_{seed}")
                print(f"\n=== {arm} / seed_{seed} ===")
                metrics_by_t, flux_by_t = evaluate_checkpoint(
                    config_path, args.gt_path, args.data_path, out, args.device
                )

                for t_val, m in metrics_by_t.items():
                    summary_rows.append({
                        "arm": arm,
                        "seed": seed,
                        "timepoint": t_val,
                        "flux_pearson_r": flux_by_t[t_val],
                        **m,
                    })

        # write sweep summary
        if summary_rows:
            summary_path = os.path.join(args.output_dir, "sweep_summary.csv")
            os.makedirs(args.output_dir, exist_ok=True)
            with open(summary_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
                writer.writeheader()
                writer.writerows(summary_rows)
            print(f"\nSweep summary: {summary_path}")
    else:
        print("Provide either --config_path (single) or --sweep_root (sweep). See --help.")
        sys.exit(1)


if __name__ == "__main__":
    main()
