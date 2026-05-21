import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

REPO_ROOT = Path("/rds/user/wz369/hpc-work/pseudodynamics_plus")
LOGS_DIR = REPO_ROOT / "logs"
FIGURE_DIR = REPO_ROOT / "figures" / "ablation"

LOSS_TERMS = {
    "lambdaD":         {"label": r"$\lambda_D$",        "values": [0, 0.01, 1, 10]},
    "lambdav":         {"label": r"$\lambda_v$",        "values": [0, 0.01, 1, 10]},
    "lambdag":         {"label": r"$\lambda_g$",        "values": [0, 0.01, 1, 10]},
    "lambdaCFM":       {"label": r"$\lambda_{CFM}$",    "values": [0, 0.01, 1, 10]},
    "lambdaNeuralODE": {"label": r"$\lambda_{ODE}$",    "values": [0, 0.01, 1, 10]},
    "lambdaR":         {"label": r"$\lambda_R$",        "values": [0, 1, 100, 1000]},
}

DATASETS = ["klein", "tom_pos"]
SEEDS = [0, 1, 2]

DATASET_COLORS = {"klein": "#1f77b4", "tom_pos": "#d62728"}
DATASET_LABELS = {"klein": "Klein (SC)", "tom_pos": "Tom+"}


def load_metric_table(dataset, arm, value, metric_name):
    # TODO: fill in once checkpoints exist
    # Expected path pattern:
    #   logs/<dataset>_ablation/<arm>_<value>/seed_<s>/eval_combined.csv
    # For each seed, load the CSV and extract the `metric_name` column (sde row).
    return np.full(len(SEEDS), np.nan)


def _collect_arm_metrics(metric_name):
    arm_data = {}
    for arm, info in LOSS_TERMS.items():
        vals = info["values"]
        rows = {}
        for ds in DATASETS:
            mat = np.full((len(vals), len(SEEDS)), np.nan)
            for vi, v in enumerate(vals):
                mat[vi] = load_metric_table(ds, arm, v, metric_name)
            rows[ds] = mat
        arm_data[arm] = {"values": vals, "per_dataset": rows}
    return arm_data


def _plot_sensitivity_row(ax_row, arm_data, metric_name, y_label):
    for col_idx, (arm, info) in enumerate(LOSS_TERMS.items()):
        ax = ax_row[col_idx]
        vals = np.array(info["values"], dtype=float)
        x_plot = np.where(vals == 0, 1e-3, vals)

        for ds in DATASETS:
            mat = arm_data[arm]["per_dataset"][ds]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                means = np.nanmean(mat, axis=1)
                sems = np.nanstd(mat, axis=1) / np.sqrt(len(SEEDS))

            # placeholder: draw flat random line so figure is not blank
            # TODO: fill in once checkpoints exist — remove these two lines
            means = np.random.rand(len(vals)) * 0.5 + 0.3
            sems = np.random.rand(len(vals)) * 0.05

            ax.errorbar(
                x_plot, means, yerr=sems,
                marker="o", ms=4, lw=1.5,
                color=DATASET_COLORS[ds],
                label=DATASET_LABELS[ds],
                capsize=3,
            )

        ax.set_xscale("log")
        ax.set_xticks(x_plot)
        ax.set_xticklabels([str(v) for v in vals], fontsize=6, rotation=45)
        ax.set_xlabel(info["label"], fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        if col_idx == 0:
            ax.set_ylabel(y_label, fontsize=8)
        else:
            ax.set_ylabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_row[0].legend(fontsize=6, loc="upper right", frameon=False)


def panel_A_sensitivity_W2(ax_row):
    arm_data = _collect_arm_metrics("w2_scaled")
    _plot_sensitivity_row(ax_row, arm_data, "w2_scaled", "W2 (scaled)")
    for col_idx, arm in enumerate(LOSS_TERMS):
        ax_row[col_idx].set_title(LOSS_TERMS[arm]["label"], fontsize=9)


def panel_B_sensitivity_fate(ax_row):
    arm_data = _collect_arm_metrics("pearson_r")
    _plot_sensitivity_row(ax_row, arm_data, "pearson_r", "Fate Pearson r")


def panel_C_cell_mass(ax):
    # TODO: fill in once checkpoints exist
    # Load bone-marrow adata, compute cell-mass per timepoint under
    # full model vs lambdaG=0 run, and plot as bar or line comparison.
    timepoints = [0, 2, 4, 6, 8]
    full_mass = np.random.rand(len(timepoints)) * 0.4 + 0.6
    no_g_mass = np.random.rand(len(timepoints)) * 0.4 + 0.6

    x = np.arange(len(timepoints))
    w = 0.35
    ax.bar(x - w / 2, full_mass, width=w, color="#4878cf", label=r"Full ($\lambda_g > 0$)")
    ax.bar(x + w / 2, no_g_mass, width=w, color="#ee854a", label=r"$\lambda_g = 0$")
    ax.set_xticks(x)
    ax.set_xticklabels([f"t={t}" for t in timepoints], fontsize=7)
    ax.set_ylabel("Relative cell mass", fontsize=8)
    ax.set_title("C  Cell-mass change (bone marrow)", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_D_diffusion_field(ax):
    # TODO: fill in once checkpoints exist
    # Load Klein adata + trained model, compute D field on UMAP grid,
    # plot as scatter/imshow with branching annotations.
    rng = np.random.default_rng(42)
    n_cells = 400
    umap1 = rng.normal(0, 1, n_cells)
    umap2 = rng.normal(0, 1, n_cells)
    d_field = rng.uniform(0.05, 1.5, n_cells)

    sc = ax.scatter(umap1, umap2, c=d_field, cmap="YlOrRd", s=8, alpha=0.8)
    plt.colorbar(sc, ax=ax, shrink=0.7, label="D (a.u.)")

    # placeholder branch annotations
    for label, xy in [("HSC", (-0.5, 0.8)), ("Ery", (1.2, -0.5)), ("MK", (-1.1, -0.8))]:
        ax.annotate(label, xy=xy, fontsize=7, color="navy",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="navy", lw=0.8))

    ax.set_xlabel("UMAP-1", fontsize=8)
    ax.set_ylabel("UMAP-2", fontsize=8)
    ax.set_title("D  Diffusion field — Klein", fontsize=9, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_supp_fig_1(output_path):
    n_arms = len(LOSS_TERMS)

    fig = plt.figure(figsize=(3 * n_arms, 10))
    fig.suptitle("Supp. Figure 1 — Real-data sensitivity", fontsize=11, y=0.98)

    outer = gridspec.GridSpec(3, 1, figure=fig,
                              height_ratios=[1, 1, 1.2],
                              hspace=0.55)

    # Rows A and B: one subplot per loss-term column
    row_A_gs = gridspec.GridSpecFromSubplotSpec(1, n_arms,
                                                subplot_spec=outer[0],
                                                wspace=0.45)
    row_B_gs = gridspec.GridSpecFromSubplotSpec(1, n_arms,
                                                subplot_spec=outer[1],
                                                wspace=0.45)

    ax_A = [fig.add_subplot(row_A_gs[0, c]) for c in range(n_arms)]
    ax_B = [fig.add_subplot(row_B_gs[0, c]) for c in range(n_arms)]

    panel_A_sensitivity_W2(ax_A)
    panel_A_sensitivity_W2.__doc__  # suppress unused warning
    ax_A[0].annotate("A", xy=(-0.25, 1.1), xycoords="axes fraction",
                     fontsize=11, fontweight="bold")

    panel_B_sensitivity_fate(ax_B)
    ax_B[0].annotate("B", xy=(-0.25, 1.1), xycoords="axes fraction",
                     fontsize=11, fontweight="bold")

    # Row C + D side by side
    bottom_gs = gridspec.GridSpecFromSubplotSpec(1, 2,
                                                 subplot_spec=outer[2],
                                                 wspace=0.4)
    ax_C = fig.add_subplot(bottom_gs[0, 0])
    ax_D = fig.add_subplot(bottom_gs[0, 1])

    panel_C_cell_mass(ax_C)
    ax_C.annotate("C", xy=(-0.18, 1.08), xycoords="axes fraction",
                  fontsize=11, fontweight="bold")

    panel_D_diffusion_field(ax_D)
    ax_D.annotate("D", xy=(-0.18, 1.08), xycoords="axes fraction",
                  fontsize=11, fontweight="bold")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _toy_fp_schematic(ax):
    # TODO: fill in once checkpoints exist
    # Draw a 2D double-well potential or bifurcation diagram as schematic.
    x = np.linspace(-2, 2, 200)
    V = (x ** 2 - 1) ** 2
    ax.plot(x, V, lw=2, color="steelblue")
    ax.fill_between(x, V, alpha=0.15, color="steelblue")
    ax.axvline(-1, ls="--", lw=0.8, color="gray")
    ax.axvline(1, ls="--", lw=0.8, color="gray")
    ax.annotate("FP1", xy=(-1, 0), fontsize=8, ha="center", va="top", color="navy")
    ax.annotate("FP2", xy=(1, 0), fontsize=8, ha="center", va="top", color="firebrick")
    ax.set_xlabel("state", fontsize=8)
    ax.set_ylabel("V(x)", fontsize=8)
    ax.set_title("A  Toy system (double-well)", fontsize=9, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _recovered_D_scatter(ax):
    # TODO: fill in once checkpoints exist
    # For each lambda_D value, load recovered D from checkpoint and compare
    # to ground-truth D; show scatter per lambda_D with color coding.
    rng = np.random.default_rng(0)
    lD_vals = LOSS_TERMS["lambdaD"]["values"]
    cmap = matplotlib.colormaps.get_cmap("plasma").resampled(len(lD_vals))

    for i, lD in enumerate(lD_vals):
        gt_D = rng.uniform(0.1, 2.0, 30)
        rec_D = gt_D + rng.normal(0, 0.3 * (1 + i * 0.3), 30)
        ax.scatter(gt_D, rec_D, s=18, alpha=0.7, color=cmap(i),
                   label=rf"$\lambda_D$={lD}")

    lim_max = 3.0
    ax.plot([0, lim_max], [0, lim_max], "k--", lw=0.8, label="y=x")
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_xlabel("Ground-truth D", fontsize=8)
    ax.set_ylabel("Recovered D", fontsize=8)
    ax.set_title(r"B  Recovered D vs GT per $\lambda_D$", fontsize=9, loc="left")
    ax.legend(fontsize=6, frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _recovery_metrics_curves(ax):
    # TODO: fill in once checkpoints exist
    # Plot R^2 / RMSE vs lambda_D for diffusion recovery.
    rng = np.random.default_rng(1)
    lD_vals = np.array(LOSS_TERMS["lambdaD"]["values"], dtype=float)
    x_plot = np.where(lD_vals == 0, 1e-3, lD_vals)

    r2_mean = np.array([0.10, 0.45, 0.80, 0.65]) + rng.normal(0, 0.03, 4)
    rmse_mean = np.array([1.5, 0.9, 0.4, 0.55]) + rng.normal(0, 0.05, 4)

    ax2 = ax.twinx()
    ax.plot(x_plot, r2_mean, "o-", color="#2ca02c", lw=1.5, ms=5, label=r"$R^2$")
    ax2.plot(x_plot, rmse_mean, "s--", color="#9467bd", lw=1.5, ms=5, label="RMSE")

    ax.set_xscale("log")
    ax.set_xticks(x_plot)
    ax.set_xticklabels([str(v) for v in lD_vals], fontsize=7)
    ax.set_xlabel(r"$\lambda_D$", fontsize=8)
    ax.set_ylabel(r"$R^2$ (D recovery)", fontsize=8, color="#2ca02c")
    ax2.set_ylabel("RMSE (D recovery)", fontsize=8, color="#9467bd")
    ax.set_title(r"C  Recovery metrics vs $\lambda_D$", fontsize=9, loc="left")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, frameon=False)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)


def _drift_growth_recovery(ax):
    # TODO: fill in once checkpoints exist
    # Control panel: show drift (v) and growth (g) recovery are robust across lambda_D.
    rng = np.random.default_rng(2)
    lD_vals = np.array(LOSS_TERMS["lambdaD"]["values"], dtype=float)
    x_plot = np.where(lD_vals == 0, 1e-3, lD_vals)

    v_r2 = np.array([0.78, 0.80, 0.81, 0.79]) + rng.normal(0, 0.02, 4)
    g_r2 = np.array([0.55, 0.57, 0.56, 0.58]) + rng.normal(0, 0.02, 4)

    ax.plot(x_plot, v_r2, "o-", color="#1f77b4", lw=1.5, ms=5, label=r"$R^2$ drift (v)")
    ax.plot(x_plot, g_r2, "s--", color="#ff7f0e", lw=1.5, ms=5, label=r"$R^2$ growth (g)")

    ax.set_xscale("log")
    ax.set_xticks(x_plot)
    ax.set_xticklabels([str(v) for v in lD_vals], fontsize=7)
    ax.set_xlabel(r"$\lambda_D$", fontsize=8)
    ax.set_ylabel(r"$R^2$", fontsize=8)
    ax.set_title(r"D  Drift & growth recovery (control)", fontsize=9, loc="left")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_supp_fig_2(output_path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Supp. Figure 2 — Synthetic FP recovery", fontsize=11, y=0.98)

    ax_A, ax_B, ax_C, ax_D = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    _toy_fp_schematic(ax_A)
    ax_A.annotate("A", xy=(-0.15, 1.08), xycoords="axes fraction",
                  fontsize=11, fontweight="bold")

    _recovered_D_scatter(ax_B)
    ax_B.annotate("B", xy=(-0.15, 1.08), xycoords="axes fraction",
                  fontsize=11, fontweight="bold")

    _recovery_metrics_curves(ax_C)
    ax_C.annotate("C", xy=(-0.15, 1.08), xycoords="axes fraction",
                  fontsize=11, fontweight="bold")

    _drift_growth_recovery(ax_D)
    ax_D.annotate("D", xy=(-0.15, 1.08), xycoords="axes fraction",
                  fontsize=11, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    p = argparse.ArgumentParser(description="Produce ablation supplementary figures.")
    p.add_argument("--fig", choices=["1", "2", "both"], default="both")
    p.add_argument("--output_dir", default=str(FIGURE_DIR))
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.fig in {"1", "both"}:
        make_supp_fig_1(Path(args.output_dir) / "supp_fig_1_sensitivity.png")
    if args.fig in {"2", "both"}:
        make_supp_fig_2(Path(args.output_dir) / "supp_fig_2_synthetic.png")


if __name__ == "__main__":
    main()
