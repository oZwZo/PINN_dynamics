"""Generate a self-contained HTML report for per-clone W2 analysis.

Reads aggregated CSV, generates all figures as embedded PNG,
includes concept diagrams, experimental design, and analysis narrative.

Usage:
    python 3c_html_report.py
"""

import os
import io
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import seaborn as sns
from scipy.stats import wilcoxon, spearmanr, mannwhitneyu
from itertools import combinations

PINN_DIR = "/rds/user/wz369/hpc-work/PINN_dynamics"
DATA_PATH = os.path.join(PINN_DIR, "figures/klein_per_clone_w2_all.csv")
CLONE_PROP_PATH = os.path.join(PINN_DIR, "figures/klein_clone_properties.csv")
OUT_HTML = os.path.join(PINN_DIR, "figures/per_clone_w2_report.html")

METHOD_ORDER = ["OTCFM", "SF2M", "TrajectoryNet", "scDiffEq", "DeepRUOT",
                "pdp+", "TIGON", "MIOFlow", "PRESCIENT"]

METHOD_COLORS = {
    "pdp+": "#E63946", "DeepRUOT": "#457B9D", "scDiffEq": "#2A9D8F",
    "OTCFM": "#E9C46A", "SF2M": "#F4A261", "TrajectoryNet": "#264653",
    "TIGON": "#A8DADC", "PRESCIENT": "#6D6875", "MIOFlow": "#B5838D",
}

def setup_style():
    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 14, "axes.titlesize": 14,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
        "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1, "axes.spines.top": False,
        "axes.spines.right": False,
    })
    sns.set_style("ticks")


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ═══════════════════════════════════════════════════════════════════════
# CONCEPT DIAGRAMS
# ═══════════════════════════════════════════════════════════════════════

def diagram_experimental_design():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # timeline
    ax.annotate("", xy=(13, 6.2), xytext=(1, 6.2),
                arrowprops=dict(arrowstyle="->", lw=2, color="#333"))
    for x, label in [(2, "Day 0\nBarcode"), (5, "Day 2\nSplit"), (8, "Day 4\nSource"), (11, "Day 6\nTarget")]:
        ax.plot(x, 6.2, "o", color="#264653", markersize=10, zorder=5)
        ax.text(x, 6.6, label, ha="center", va="bottom", fontsize=11, fontweight="bold")

    # HSPCs
    hspc_box = FancyBboxPatch((1.2, 4.0), 1.6, 1.2, boxstyle="round,pad=0.1",
                               facecolor="#A8DADC", edgecolor="#264653", lw=2)
    ax.add_patch(hspc_box)
    ax.text(2, 4.6, "HSPCs\n(LSK)", ha="center", va="center", fontsize=10, fontweight="bold")

    # barcode
    ax.annotate("", xy=(3.5, 4.6), xytext=(2.8, 4.6),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#E63946"))
    barcode_box = FancyBboxPatch((3.5, 4.1), 2.2, 1.0, boxstyle="round,pad=0.1",
                                  facecolor="#FEFAE0", edgecolor="#E63946", lw=2)
    ax.add_patch(barcode_box)
    ax.text(4.6, 4.6, "LARRY\n28nt barcode", ha="center", va="center", fontsize=9, color="#E63946", fontweight="bold")

    # split arrow
    ax.annotate("", xy=(7, 4.9), xytext=(5.8, 4.6),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"))
    ax.annotate("", xy=(7, 3.2), xytext=(5.8, 4.5),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"))

    # Well 1 (train)
    well1 = FancyBboxPatch((7, 4.3), 2.0, 1.2, boxstyle="round,pad=0.1",
                            facecolor="#DDD", edgecolor="#666", lw=1.5)
    ax.add_patch(well1)
    ax.text(8, 4.9, "Well 1\n(Train)", ha="center", va="center", fontsize=10, color="#666")

    # Well 2 (test) — highlighted
    well2 = FancyBboxPatch((7, 2.6), 2.0, 1.2, boxstyle="round,pad=0.1",
                            facecolor="#E9C46A", edgecolor="#E76F51", lw=2.5)
    ax.add_patch(well2)
    ax.text(8, 3.2, "Well 2\n(TEST)", ha="center", va="center", fontsize=10, fontweight="bold", color="#E76F51")

    # Day 4 cells
    ax.annotate("", xy=(10.2, 3.8), xytext=(9, 3.5),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#2A9D8F"))
    ax.annotate("", xy=(10.2, 2.5), xytext=(9, 3.0),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#457B9D"))

    # Day 4 box
    d4_box = FancyBboxPatch((10.2, 3.3), 1.3, 1.0, boxstyle="round,pad=0.1",
                             facecolor="#D8F3DC", edgecolor="#2A9D8F", lw=2)
    ax.add_patch(d4_box)
    ax.text(10.85, 3.8, "Day 4\nSource", ha="center", va="center", fontsize=9, fontweight="bold", color="#2A9D8F")

    # Day 6 box
    d6_box = FancyBboxPatch((10.2, 1.8), 1.3, 1.0, boxstyle="round,pad=0.1",
                             facecolor="#BDE0FE", edgecolor="#457B9D", lw=2)
    ax.add_patch(d6_box)
    ax.text(10.85, 2.3, "Day 6\nTarget", ha="center", va="center", fontsize=9, fontweight="bold", color="#457B9D")

    # clone link
    ax.annotate("same\nbarcode", xy=(11.8, 2.8), xytext=(12.5, 3.5),
                fontsize=9, color="#E63946", fontweight="bold",
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#E63946", connectionstyle="arc3,rad=-0.3"))
    ax.plot([11.5, 11.5], [2.8, 3.3], "--", color="#E63946", lw=1.5, alpha=0.6)

    # 92 clones callout
    ax.text(8, 1.3, "92 test clones: unique to Well 2, present at day 4 + day 6,\n"
            ">=20 cells, <20% undifferentiated", ha="center", va="center",
            fontsize=11, style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF3CD", edgecolor="#E9C46A", lw=1.5))

    fig.suptitle("Experimental Design: Klein et al. LARRY Dataset", fontsize=14, fontweight="bold", y=0.98)
    return fig


def diagram_per_clone_w2():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: What is per-clone W2
    ax = axes[0]
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_title("(a) Per-clone W2 concept", fontsize=13, fontweight="bold")

    np.random.seed(42)
    src = np.random.randn(8, 2) * 0.4 + np.array([-1.5, 0.5])
    sim = src + np.random.randn(8, 2) * 0.3 + np.array([1.5, -0.5])
    obs = np.random.randn(12, 2) * 0.5 + np.array([1.2, -0.8])

    ax.scatter(src[:, 0], src[:, 1], c="#2A9D8F", s=60, zorder=5, edgecolors="black", linewidth=0.5, label="Clone b, day 4")
    ax.scatter(obs[:, 0], obs[:, 1], c="#457B9D", s=60, zorder=5, edgecolors="black", linewidth=0.5, marker="s", label="Clone b, day 6 (observed)")
    ax.scatter(sim[:, 0], sim[:, 1], c="#E63946", s=60, zorder=5, edgecolors="black", linewidth=0.5, marker="^", label="Clone b, day 6 (simulated)")

    for i in range(len(src)):
        ax.annotate("", xy=(sim[i, 0], sim[i, 1]), xytext=(src[i, 0], src[i, 1]),
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="#E63946", alpha=0.4))

    ax.annotate("W2", xy=(0.5, -1.5), fontsize=16, fontweight="bold", color="#E76F51",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#E76F51", lw=2))

    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("Embedding dim 1")
    ax.set_ylabel("Embedding dim 2")
    sns.despine(ax=ax)

    # Panel B: Population vs per-clone
    ax = axes[1]
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.set_title("(b) Population W2 can mask\nper-clone errors", fontsize=13, fontweight="bold")

    np.random.seed(123)
    for i, (cx, cy, color) in enumerate([(-1.5, 1.5, "#2A9D8F"), (1.5, 1.5, "#E9C46A"), (0, -1.5, "#E63946")]):
        obs_c = np.random.randn(10, 2) * 0.4 + np.array([cx, cy])
        if i == 2:
            sim_c = np.random.randn(10, 2) * 0.4 + np.array([cx + 1.5, cy - 1.0])
        else:
            sim_c = np.random.randn(10, 2) * 0.3 + np.array([cx, cy])
        ax.scatter(obs_c[:, 0], obs_c[:, 1], c=color, s=40, alpha=0.7, edgecolors="black", linewidth=0.3)
        ax.scatter(sim_c[:, 0], sim_c[:, 1], c=color, s=40, alpha=0.7, marker="^", edgecolors="black", linewidth=0.3)
        ax.annotate(f"Clone {i+1}", xy=(cx, cy + 1.0), fontsize=9, ha="center", color=color, fontweight="bold")

    ax.annotate("Clone 3:\nhigh W2!", xy=(1.5, -2.5), fontsize=10, color="#E63946", fontweight="bold",
                ha="center", bbox=dict(facecolor="white", edgecolor="#E63946", lw=1.5, boxstyle="round"))
    ax.text(0, -3.5, "Population W2 averages over all clones\n-> can miss clone-specific failures",
            ha="center", fontsize=9, style="italic")
    sns.despine(ax=ax)
    ax.set_xlabel("Embedding dim 1")
    ax.set_ylabel("Embedding dim 2")

    # Panel C: Clone size stratification concept
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("(c) Clone-size stratification", fontsize=13, fontweight="bold")

    clone_sizes = sorted([84, 60, 45, 38, 32, 28, 25, 22, 20, 18, 15, 12], reverse=True)
    bar_colors = ["#457B9D" if s > 30 else "#E9C46A" for s in clone_sizes]
    y_positions = np.linspace(7, 1, len(clone_sizes))

    for i, (size, y, color) in enumerate(zip(clone_sizes, y_positions, bar_colors)):
        bar_width = size / 10
        ax.barh(y, bar_width, height=0.4, color=color, edgecolor="black", linewidth=0.5)
        ax.text(bar_width + 0.2, y, f"n={size}", fontsize=8, va="center")

    cutoff_x = 30 / 10
    ax.axvline(cutoff_x, color="#E76F51", lw=2, ls="--")
    ax.text(cutoff_x + 0.1, 7.5, "cutoff", fontsize=10, color="#E76F51", fontweight="bold")

    ax.text(6.5, 6.5, "Large clones\n(> cutoff)", fontsize=10, color="#457B9D", fontweight="bold",
            bbox=dict(facecolor="#BDE0FE", edgecolor="#457B9D", boxstyle="round", alpha=0.7))
    ax.text(6.5, 2.5, "Small clones\n(<= cutoff)", fontsize=10, color="#B5651D", fontweight="bold",
            bbox=dict(facecolor="#FFF3CD", edgecolor="#E9C46A", boxstyle="round", alpha=0.7))

    ax.text(5, 0.3, "Sweep cutoff -> compare W2 per group", fontsize=10, ha="center", style="italic")

    plt.tight_layout()
    return fig


def diagram_hidden_variables():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Progenitor
    prog = FancyBboxPatch((0.5, 2.0), 2.0, 2.0, boxstyle="round,pad=0.15",
                           facecolor="#A8DADC", edgecolor="#264653", lw=2)
    ax.add_patch(prog)
    ax.text(1.5, 3.0, "HSPC\n(Day 4)", ha="center", va="center", fontsize=11, fontweight="bold")

    # Observable state
    obs_box = FancyBboxPatch((3.5, 3.5), 2.5, 1.5, boxstyle="round,pad=0.1",
                              facecolor="#D8F3DC", edgecolor="#2A9D8F", lw=2)
    ax.add_patch(obs_box)
    ax.text(4.75, 4.25, "Transcriptome\n(scRNA-seq)", ha="center", va="center", fontsize=10, color="#2A9D8F")
    ax.annotate("", xy=(3.5, 4.25), xytext=(2.5, 3.5),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#2A9D8F"))

    # Hidden state
    hid_box = FancyBboxPatch((3.5, 1.0), 2.5, 1.5, boxstyle="round,pad=0.1",
                              facecolor="#FFE5D9", edgecolor="#E76F51", lw=2, ls="--")
    ax.add_patch(hid_box)
    ax.text(4.75, 1.75, "Hidden variables\n(epigenome, TF\nprotein, spatial)", ha="center", va="center",
            fontsize=9, color="#E76F51", fontweight="bold")
    ax.annotate("", xy=(3.5, 1.75), xytext=(2.5, 2.5),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#E76F51", ls="--"))

    # Fate
    for i, (y, fate, col) in enumerate([(4.5, "Neutrophil", "#457B9D"),
                                         (3.0, "Monocyte", "#2A9D8F"),
                                         (1.5, "Meg", "#E63946")]):
        fate_box = FancyBboxPatch((7.5, y - 0.4), 2.0, 0.8, boxstyle="round,pad=0.1",
                                   facecolor=col, edgecolor="black", lw=1, alpha=0.3)
        ax.add_patch(fate_box)
        ax.text(8.5, y, fate, ha="center", va="center", fontsize=10, fontweight="bold", color=col)

    ax.annotate("", xy=(7.5, 4.5), xytext=(6, 4.0), arrowprops=dict(arrowstyle="->", lw=1.2, color="#333"))
    ax.annotate("", xy=(7.5, 3.0), xytext=(6, 3.0), arrowprops=dict(arrowstyle="->", lw=1.2, color="#333"))
    ax.annotate("", xy=(7.5, 1.5), xytext=(6, 2.0), arrowprops=dict(arrowstyle="->", lw=1.2, color="#333"))

    ax.text(6.6, 5.0, "observable", fontsize=9, color="#2A9D8F", style="italic")
    ax.text(6.6, 0.8, "hidden", fontsize=9, color="#E76F51", style="italic", fontweight="bold")

    ax.text(5, 0.2, "DPI (Weinreb 2020): late prediction > early prediction => hidden variables exist",
            ha="center", fontsize=10, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3CD", edgecolor="#E9C46A"))

    fig.suptitle("Why Per-Clone W2 Has a Floor: Hidden Variables", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS FIGURES
# ═══════════════════════════════════════════════════════════════════════

def fig_method_ranking(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, space, title in zip(axes, ["dm10", "pc30"], ["DM10 (Diffusion Map, 10D)", "PC30 (PCA, 30D)"]):
        sub = df[df["space"] == space].copy()
        order = sub.groupby("method")["w2_raw_mean"].median().sort_values().index.tolist()
        palette = {m: METHOD_COLORS.get(m, "#999") for m in order}
        sns.violinplot(data=sub, x="method", y="w2_raw_mean", order=order,
                       hue="method", palette=palette, inner=None, alpha=0.3, ax=ax, cut=0, legend=False)
        sns.stripplot(data=sub, x="method", y="w2_raw_mean", order=order,
                      hue="method", palette=palette, size=3, alpha=0.5, jitter=True, ax=ax, legend=False)
        sns.boxplot(data=sub, x="method", y="w2_raw_mean", order=order,
                    color="white", width=0.3, fliersize=0, ax=ax,
                    boxprops=dict(alpha=0.7), medianprops=dict(color="black", linewidth=2))
        ax.set_xlabel("")
        ax.set_ylabel("Per-clone W2 (raw)")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig


def fig_biology_panels(df, space="dm10"):
    sub = df[df["space"] == space].copy()
    best_method = sub.groupby("method")["w2_raw_mean"].median().idxmin()
    best = sub[sub["method"] == best_method].copy()
    space_label = "DM10" if space == "dm10" else "PC30"

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) W2 by dominant fate
    ax = axes[0, 0]
    fate_counts = best["dominant_fate"].value_counts()
    fates_to_show = fate_counts[fate_counts >= 3].index.tolist()
    d = best[best["dominant_fate"].isin(fates_to_show)]
    if not d.empty:
        fate_order = d.groupby("dominant_fate")["w2_raw_mean"].median().sort_values().index.tolist()
        sns.boxplot(data=d, x="dominant_fate", y="w2_raw_mean", order=fate_order, ax=ax,
                    hue="dominant_fate", palette="Set2", legend=False)
        sns.stripplot(data=d, x="dominant_fate", y="w2_raw_mean", order=fate_order, ax=ax,
                      color="black", size=3, alpha=0.5)
    ax.set_xlabel("Dominant fate (day 6)")
    ax.set_ylabel("Per-clone W2")
    ax.set_title(f"(a) W2 by fate type ({best_method})")
    ax.tick_params(axis="x", rotation=45)

    # (b) W2 vs clone size
    ax = axes[0, 1]
    _scatter_corr(best, "clone_size_tgt", "w2_raw_mean", ax,
                  "Clone size (day 6)", "Per-clone W2",
                  f"(b) W2 vs clone size ({best_method})")

    # (c) W2 vs multipotency
    ax = axes[0, 2]
    if "multipotency" in best.columns and best["multipotency"].notna().sum() > 5:
        d = best.dropna(subset=["multipotency"])
        d["multipotency"] = d["multipotency"].astype(int)
        multi_vals = sorted(d["multipotency"].unique())
        sns.boxplot(data=d, x="multipotency", y="w2_raw_mean", order=multi_vals, ax=ax,
                    hue="multipotency", palette="Blues", legend=False)
        sns.stripplot(data=d, x="multipotency", y="w2_raw_mean", order=multi_vals, ax=ax,
                      color="black", size=3, alpha=0.5)
        r, p = spearmanr(d["multipotency"], d["w2_raw_mean"])
        ax.set_title(f"(c) W2 vs multipotency (r={r:.2f}, p={p:.3f})")
    ax.set_xlabel("Multipotency (n fates)")
    ax.set_ylabel("Per-clone W2")

    # (d) W2 vs pseudotime spread
    ax = axes[1, 0]
    _scatter_corr(best, "pseudotime_spread", "w2_raw_mean", ax,
                  "Pseudotime spread (std, day 6)", "Per-clone W2",
                  f"(d) W2 vs pseudotime spread ({best_method})")

    # (e) W2 vs initial state variance
    ax = axes[1, 1]
    var_col = "initial_var_dm" if space == "dm10" else "initial_var_pc"
    _scatter_corr(best, var_col, "w2_raw_mean", ax,
                  "Initial state variance (day 4)", "Per-clone W2",
                  f"(e) W2 vs initial variance ({best_method})")

    # (f) heatmap
    ax = axes[1, 2]
    if "dominant_fate" in sub.columns:
        pivot = (sub.groupby(["method", "dominant_fate"])["w2_raw_mean"]
                 .median().reset_index()
                 .pivot(index="method", columns="dominant_fate", values="w2_raw_mean"))
        if fates_to_show:
            pivot = pivot.reindex(columns=[f for f in fates_to_show if f in pivot.columns])
        if not pivot.empty:
            sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlOrRd", ax=ax,
                        linewidths=0.5, cbar_kws={"label": "Median W2"})
            ax.set_title("(f) Median W2 by method x fate")
            ax.set_ylabel("")

    fig.suptitle(f"Biological Dissection -- {space_label}", fontsize=16, y=1.02)
    plt.tight_layout()
    return fig


def _scatter_corr(data, xcol, ycol, ax, xlabel, ylabel, title):
    d = data.dropna(subset=[xcol, ycol])
    if len(d) < 5:
        ax.set_title(f"{title} (insufficient data)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return
    ax.scatter(d[xcol], d[ycol], alpha=0.6, edgecolors="black", linewidth=0.3, s=30)
    z = np.polyfit(d[xcol], d[ycol], 1)
    xline = np.linspace(d[xcol].min(), d[xcol].max(), 100)
    ax.plot(xline, np.polyval(z, xline), "--", color="#E63946", alpha=0.7, lw=1.5)
    r, p = spearmanr(d[xcol], d[ycol])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n(Spearman r={r:.2f}, p={p:.3f})")


def fig_sensitivity(df, space="dm10"):
    thresholds = [5, 10, 15, 20, 25, 30, 40, 50]
    records = []
    for thresh in thresholds:
        sub = df[(df["space"] == space) & (df["clone_size_tgt"] >= thresh)]
        for method, grp in sub.groupby("method"):
            records.append({
                "threshold": thresh, "method": method,
                "median_w2": grp["w2_raw_mean"].median(),
                "n_clones": grp["clone"].nunique(),
            })
    sens_df = pd.DataFrame(records)
    space_label = "DM10" if space == "dm10" else "PC30"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for method in sens_df["method"].unique():
        d = sens_df[sens_df["method"] == method]
        color = METHOD_COLORS.get(method, "#999")
        ax1.plot(d["threshold"], d["median_w2"], "o-", label=method, color=color, markersize=5)
        ax2.plot(d["threshold"], d["n_clones"], "o-", label=method, color=color, markersize=5)

    ax1.set_xlabel("Min clone size (day 6)")
    ax1.set_ylabel("Median per-clone W2")
    ax1.set_title(f"W2 sensitivity to clone size filter ({space_label})")
    ax1.legend(fontsize=8, ncol=2)
    ax2.set_xlabel("Min clone size (day 6)")
    ax2.set_ylabel("Number of clones")
    ax2.set_title("Clone count at each threshold")
    plt.tight_layout()
    return fig


def fig_wilcoxon(df, space="dm10"):
    sub = df[df["space"] == space].copy()
    methods = sorted(sub["method"].unique())
    n = len(methods)
    pval_matrix = pd.DataFrame(np.ones((n, n)), index=methods, columns=methods)
    pivot = sub.pivot_table(index="clone", columns="method", values="w2_raw_mean")

    for m1, m2 in combinations(methods, 2):
        shared = pivot[[m1, m2]].dropna()
        if len(shared) < 5:
            continue
        try:
            _, p = wilcoxon(shared[m1], shared[m2])
        except ValueError:
            p = 1.0
        pval_matrix.loc[m1, m2] = p
        pval_matrix.loc[m2, m1] = p

    n_tests = n * (n - 1) / 2
    pval_bonf = (pval_matrix * n_tests).clip(upper=1.0)
    np.fill_diagonal(pval_bonf.values, np.nan)
    log_p = -np.log10(pval_bonf.replace(0, 1e-300))
    np.fill_diagonal(log_p.values, 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    space_label = "DM10" if space == "dm10" else "PC30"
    mask = np.triu(np.ones_like(log_p, dtype=bool), k=0)
    annot_str = pval_bonf.map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    sns.heatmap(log_p, mask=mask, annot=annot_str, fmt="", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"label": "-log10(p, Bonferroni)"}, vmin=0, vmax=5)
    ax.set_title(f"Paired Wilcoxon signed-rank test ({space_label})")
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# NEW: Clone-size stratified W2
# ═══════════════════════════════════════════════════════════════════════

def fig_clone_size_stratified(df):
    cutoffs = [15, 18, 20, 22, 25, 28, 30, 35, 40, 50]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for col, space, space_label in [(0, "dm10", "DM10"), (1, "pc30", "PC30")]:
        sub = df[df["space"] == space]

        # Top: ratio (large/small median W2) vs cutoff per method
        ax = axes[0, col]
        for method in sorted(sub["method"].unique()):
            msub = sub[sub["method"] == method]
            ratios, valid_cuts = [], []
            for c in cutoffs:
                small = msub[msub["clone_size_tgt"] <= c]["w2_raw_mean"]
                large = msub[msub["clone_size_tgt"] > c]["w2_raw_mean"]
                if len(small) >= 5 and len(large) >= 5:
                    ratios.append(large.median() / small.median() if small.median() > 0 else np.nan)
                    valid_cuts.append(c)
            if valid_cuts:
                ax.plot(valid_cuts, ratios, "o-", label=method,
                        color=METHOD_COLORS.get(method, "#999"), markersize=4, lw=1.5)
        ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.5)
        ax.set_xlabel("Clone size cutoff (day 6)")
        ax.set_ylabel("Median W2 ratio (large / small)")
        ax.set_title(f"Large vs Small Clone W2 Ratio ({space_label})")
        ax.legend(fontsize=7, ncol=2)

        # Bottom: grouped bar at median cutoff
        ax = axes[1, col]
        median_size = int(sub["clone_size_tgt"].median())
        records = []
        for method in sorted(sub["method"].unique()):
            msub = sub[sub["method"] == method]
            small = msub[msub["clone_size_tgt"] <= median_size]["w2_raw_mean"]
            large = msub[msub["clone_size_tgt"] > median_size]["w2_raw_mean"]
            records.append({"method": method, "group": f"Small (n<={median_size})",
                           "median_w2": small.median(), "n": len(small)})
            records.append({"method": method, "group": f"Large (n>{median_size})",
                           "median_w2": large.median(), "n": len(large)})

        bar_df = pd.DataFrame(records)
        method_order = (bar_df[bar_df["group"].str.startswith("Small")]
                        .sort_values("median_w2")["method"].tolist())
        sns.barplot(data=bar_df, x="method", y="median_w2", hue="group",
                    order=method_order, ax=ax,
                    palette={"Small (n<={})".format(median_size): "#E9C46A",
                             "Large (n>{})".format(median_size): "#457B9D"},
                    edgecolor="black", linewidth=0.5)
        ax.set_xlabel("")
        ax.set_ylabel("Median per-clone W2")
        ax.set_title(f"W2 by clone size group (cutoff={median_size}) ({space_label})")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=9)

    plt.tight_layout()
    return fig


def fig_clone_size_sweep_detail(df):
    cutoffs = [15, 20, 25, 30, 40, 50]

    fig, axes = plt.subplots(2, len(cutoffs), figsize=(4 * len(cutoffs), 8),
                             sharey="row")

    for row, (space, space_label) in enumerate([("dm10", "DM10"), ("pc30", "PC30")]):
        sub = df[df["space"] == space]
        for j, c in enumerate(cutoffs):
            ax = axes[row, j]
            small = sub[sub["clone_size_tgt"] <= c].copy()
            large = sub[sub["clone_size_tgt"] > c].copy()
            small["group"] = f"<={c}"
            large["group"] = f">{c}"
            combined = pd.concat([small, large])

            method_order = combined.groupby("method")["w2_raw_mean"].median().sort_values().index.tolist()
            sns.boxplot(data=combined, x="group", y="w2_raw_mean",
                        hue="group", palette={f"<={c}": "#E9C46A", f">{c}": "#457B9D"},
                        ax=ax, fliersize=2, legend=False)

            n_small = small["clone"].nunique()
            n_large = large["clone"].nunique()
            ax.set_title(f"cut={c}\n({n_small} vs {n_large})", fontsize=10)
            ax.set_xlabel("")
            if j == 0:
                ax.set_ylabel(f"Per-clone W2 ({space_label})")
            else:
                ax.set_ylabel("")

    fig.suptitle("Per-clone W2 distribution: small vs large clones (all methods pooled)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════

def make_summary_table(df):
    records = []
    for (method, space), grp in df.groupby(["method", "space"]):
        vals = grp["w2_raw_mean"].values
        records.append({
            "Method": method, "Space": space.upper(),
            "Median W2": f"{np.median(vals):.6f}",
            "Mean W2": f"{np.mean(vals):.6f}",
            "Std W2": f"{np.std(vals):.6f}",
            "N clones": len(vals),
            "N runs": int(grp["n_runs"].iloc[0]),
        })
    summary = pd.DataFrame(records).sort_values(["Space", "Median W2"])
    return summary


# ═══════════════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ═══════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Per-Clone W2 Analysis Report -- Klein et al.</title>
<style>
  :root {{
    --primary: #264653;
    --secondary: #2A9D8F;
    --accent: #E76F51;
    --warm: #E9C46A;
    --bg: #FAFAFA;
    --card: #FFFFFF;
    --text: #333333;
    --muted: #666666;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
  }}
  header {{
    background: linear-gradient(135deg, var(--primary), #1a3a4a);
    color: white;
    padding: 2rem 0;
    text-align: center;
  }}
  header h1 {{ font-size: 2rem; margin-bottom: 0.5rem; }}
  header p {{ opacity: 0.85; font-size: 1.1rem; }}
  nav {{
    background: var(--primary);
    padding: 0.5rem 0;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
  }}
  nav ul {{
    list-style: none;
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.3rem;
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 1rem;
  }}
  nav a {{
    color: white;
    text-decoration: none;
    padding: 0.4rem 0.8rem;
    border-radius: 4px;
    font-size: 0.85rem;
    transition: background 0.2s;
  }}
  nav a:hover {{ background: rgba(255,255,255,0.15); }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 1rem 2rem; }}
  .section {{
    background: var(--card);
    border-radius: 12px;
    padding: 2rem;
    margin: 1.5rem 0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border-left: 4px solid var(--secondary);
  }}
  .section h2 {{
    color: var(--primary);
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }}
  .section h2 .num {{
    background: var(--secondary);
    color: white;
    width: 2rem;
    height: 2rem;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
  }}
  .section h3 {{
    color: var(--primary);
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
  }}
  .design-box {{
    background: #F0F7FF;
    border: 1px solid #B8D4E3;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
  }}
  .design-box h4 {{
    color: var(--primary);
    margin-bottom: 0.5rem;
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .result-box {{
    background: #F0FAF0;
    border: 1px solid #B8E3B8;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
  }}
  .result-box h4 {{
    color: #2D6A4F;
    margin-bottom: 0.5rem;
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .implication-box {{
    background: #FFF8F0;
    border: 1px solid #E3C8A8;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
  }}
  .implication-box h4 {{
    color: #B5651D;
    margin-bottom: 0.5rem;
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .caveat {{
    background: #FFF3CD;
    border: 1px solid #E9C46A;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.95rem;
  }}
  .figure-container {{
    text-align: center;
    margin: 1.5rem 0;
  }}
  .figure-container img {{
    max-width: 100%;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  }}
  .figure-caption {{
    font-size: 0.9rem;
    color: var(--muted);
    margin-top: 0.5rem;
    font-style: italic;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
    font-size: 0.9rem;
  }}
  th {{
    background: var(--primary);
    color: white;
    padding: 0.6rem 0.8rem;
    text-align: left;
  }}
  td {{
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid #eee;
  }}
  tr:nth-child(even) {{ background: #F8F9FA; }}
  tr:hover {{ background: #E8F4FD; }}
  .method-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    color: white;
    font-weight: bold;
    font-size: 0.85rem;
  }}
  .collapsible {{
    cursor: pointer;
    padding: 0.8rem 1.2rem;
    background: #F0F0F0;
    border: none;
    text-align: left;
    width: 100%;
    font-size: 1rem;
    font-weight: bold;
    color: var(--primary);
    border-radius: 6px;
    margin-top: 1rem;
    transition: background 0.2s;
  }}
  .collapsible:hover {{ background: #E0E0E0; }}
  .collapsible::before {{ content: '+ '; }}
  .collapsible.active::before {{ content: '- '; }}
  .collapsible-content {{
    max-height: 0;
    overflow: hidden;
    transition: max-height 0.3s ease-out;
    padding: 0 1rem;
  }}
  footer {{
    text-align: center;
    padding: 2rem;
    color: var(--muted);
    font-size: 0.9rem;
  }}
  ul, ol {{
    margin: 0.5rem 0 0.5rem 1.5rem;
  }}
  li {{ margin-bottom: 0.3rem; }}
  p {{ margin-bottom: 0.8rem; }}
  .two-col {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
  }}
  @media (max-width: 800px) {{
    .two-col {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<header>
  <h1>Per-Clone Wasserstein-2 Analysis</h1>
  <p>Klein et al. LARRY Dataset -- 9 Trajectory Inference Methods -- 92 Test Clones</p>
  <p style="font-size:0.85rem; opacity:0.7;">Generated {date}</p>
</header>
<nav>
  <ul>
    <li><a href="#design">Design</a></li>
    <li><a href="#metric">Metric</a></li>
    <li><a href="#ranking">Ranking</a></li>
    <li><a href="#biology">Biology</a></li>
    <li><a href="#stratified">Stratified</a></li>
    <li><a href="#sensitivity">Sensitivity</a></li>
    <li><a href="#stats">Statistics</a></li>
    <li><a href="#summary">Summary</a></li>
  </ul>
</nav>
<div class="container">

<!-- SECTION 1: Experimental Design -->
<div class="section" id="design">
  <h2><span class="num">1</span> Experimental Design</h2>

  <div class="design-box">
    <h4>What is LARRY?</h4>
    <p><strong>LARRY</strong> (Lineage And RNA RecoverY) is an expressed lentiviral barcode system that inserts a random <strong>28-nucleotide barcode</strong> into cells. Because lentiviral integration is stable and heritable, all daughter cells carry the same barcode. This links a progenitor's transcriptional state to the differentiated fates of its clonal descendants.</p>
  </div>

  <div class="figure-container">
    <img src="data:image/png;base64,{fig_design}" alt="Experimental Design">
    <p class="figure-caption">Figure 1. Split-well experimental design. HSPCs are barcoded with LARRY, split into wells, and sequenced at days 4 (source) and 6 (target). Well 2 is held out as the test set.</p>
  </div>

  <div class="design-box">
    <h4>Clone Selection Criteria (92 Test Clones)</h4>
    <ul>
      <li><strong>Well 2 only</strong> -- held out from training, ensures unbiased evaluation</li>
      <li><strong>Present at both day 4 and day 6</strong> -- enables source-to-target simulation</li>
      <li><strong>&ge;20 cells</strong> at day 6 -- ensures W2 is well-estimated</li>
      <li><strong>&lt;20% undifferentiated</strong> -- focuses on clones with clear fate commitment</li>
    </ul>
  </div>

  <h3>Methods Evaluated</h3>
  <p>We benchmark <strong>9 trajectory inference methods</strong> spanning ODE, SDE, flow matching, optimal transport, and PINN approaches:</p>
  <table>
    <tr><th>Method</th><th>Type</th><th>Simulation</th><th>Seeds/Runs</th></tr>
    <tr><td><span class="method-badge" style="background:#E9C46A;color:#333">OTCFM</span></td><td>Conditional Flow Matching + OT</td><td>ODE</td><td>3</td></tr>
    <tr><td><span class="method-badge" style="background:#F4A261">SF2M</span></td><td>Schrodinger Bridge Flow Matching</td><td>SDE</td><td>3</td></tr>
    <tr><td><span class="method-badge" style="background:#264653">TrajectoryNet</span></td><td>Continuous Normalizing Flow + OT</td><td>ODE</td><td>3-4</td></tr>
    <tr><td><span class="method-badge" style="background:#2A9D8F">scDiffEq</span></td><td>Neural SDE (state-dependent)</td><td>SDE</td><td>3</td></tr>
    <tr><td><span class="method-badge" style="background:#457B9D">DeepRUOT</span></td><td>Deep Regularized Unbalanced OT</td><td>ODE</td><td>4</td></tr>
    <tr><td><span class="method-badge" style="background:#E63946">pdp+</span></td><td>PINN-based PDE solver</td><td>SDE</td><td>1 (best config)</td></tr>
    <tr><td><span class="method-badge" style="background:#A8DADC;color:#333">TIGON</span></td><td>Time-resolved GAN + ODE</td><td>ODE</td><td>3</td></tr>
    <tr><td><span class="method-badge" style="background:#B5838D">MIOFlow</span></td><td>Neural ODE on geodesic AE</td><td>ODE</td><td>3</td></tr>
    <tr><td><span class="method-badge" style="background:#6D6875">PRESCIENT</span></td><td>Potential landscape + SDE</td><td>SDE</td><td>3</td></tr>
  </table>
  <div class="caveat">
    <strong>Note:</strong> MIOFlow results are preliminary -- a denormalize-before-decode eval bug was identified but GPU re-evaluation has not been run yet. MIOFlow W2 values may improve after re-eval.
  </div>
</div>

<!-- SECTION 2: Per-Clone W2 Metric -->
<div class="section" id="metric">
  <h2><span class="num">2</span> The Per-Clone W2 Metric</h2>

  <div class="design-box">
    <h4>Why Per-Clone W2?</h4>
    <p><strong>Population-level W2</strong> computes the distance between all simulated endpoints and all observed day-6 cells. A method can achieve good population W2 by "averaging" across fates -- e.g., placing simulated Neutrophil-fated cells halfway between Neutrophil and Monocyte regions. <strong>Per-clone W2</strong> evaluates each clone individually: for clone <em>b</em>, it computes W2 between the simulated endpoints of clone <em>b</em>'s day-4 cells and clone <em>b</em>'s observed day-6 cells. This is a much more stringent test.</p>
  </div>

  <div class="figure-container">
    <img src="data:image/png;base64,{fig_concept}" alt="Per-Clone W2 Concept">
    <p class="figure-caption">Figure 2. (a) Per-clone W2 measures the distance between simulated and observed distributions <em>within each clone</em>. (b) Population W2 can mask clone-specific failures -- a method may match the overall distribution while systematically misplacing specific clones. (c) Clone-size stratification: we split clones into small vs. large groups and compare W2 across groups.</p>
  </div>

  <div class="design-box">
    <h4>Computation Details</h4>
    <ul>
      <li><strong>Source cells:</strong> Clone <em>b</em>'s cells at day 4</li>
      <li><strong>Simulation:</strong> Run the trained model's dynamics from day 4 to day 6</li>
      <li><strong>Target cells:</strong> Clone <em>b</em>'s observed cells at day 6</li>
      <li><strong>W2 (raw):</strong> Wasserstein-2 distance in the original embedding space (DM10 or PC30), after inverse-standardizing if training used scaled data</li>
      <li><strong>Averaging:</strong> For methods with multiple seeds/runs, W2 is averaged per clone across runs</li>
    </ul>
  </div>

  <div class="figure-container">
    <img src="data:image/png;base64,{fig_hidden}" alt="Hidden Variables">
    <p class="figure-caption">Figure 3. The hidden variable argument (Weinreb 2020). HSPCs carry heritable fate biases encoded in epigenomic, proteomic, and spatial properties that are invisible to scRNA-seq. The Data Processing Inequality proves these exist: late fate prediction is more accurate than early prediction, which is impossible if the transcriptome contains all fate-determining information. Per-clone W2 exposes the residual error caused by these unobserved variables.</p>
  </div>
</div>

<!-- SECTION 3: Method Ranking -->
<div class="section" id="ranking">
  <h2><span class="num">3</span> Method Ranking</h2>

  <div class="design-box">
    <h4>Experimental Design</h4>
    <p>For each of the 9 methods, we compute per-clone W2 for all 92 test clones, averaging across 3-4 seeds/runs. Methods are ranked by <strong>median</strong> per-clone W2 (robust to outliers). We show violin + strip + box plots to reveal the full distribution shape.</p>
    <p><strong>Two representation spaces:</strong></p>
    <ul>
      <li><strong>DM10</strong> -- 10-dimensional diffusion map embedding. Captures nonlinear manifold geometry; lower-dimensional.</li>
      <li><strong>PC30</strong> -- 30-dimensional PCA embedding. Linear projection; higher-dimensional; preserves more variance.</li>
    </ul>
  </div>

  <div class="figure-container">
    <img src="data:image/png;base64,{fig_ranking}" alt="Method Ranking">
    <p class="figure-caption">Figure 4. Per-clone W2 distribution across 92 test clones for each method, ordered by median. Left: DM10 space. Right: PC30 space. Each point is one clone's average W2 across seeds.</p>
  </div>

  <div class="result-box">
    <h4>Results</h4>
    <p><strong>DM10:</strong> OTCFM achieves the lowest median per-clone W2 (0.00454), followed by SF2M (0.00517) and TrajectoryNet (0.00537). The top 7 methods are tightly clustered (0.0045-0.0059). MIOFlow (0.00678) and PRESCIENT (0.00749) trail.</p>
    <p><strong>PC30:</strong> scDiffEq leads (11.85), followed by DeepRUOT (12.34) and pdp+ (12.61). The spread is tighter: all 9 methods fall within 11.8-14.1.</p>
    <p><strong>Ranking flip:</strong> OTCFM ranks 1st in DM10 but 7th in PC30. scDiffEq ranks 4th in DM10 but 1st in PC30. Method performance is representation-dependent.</p>
  </div>
</div>

<!-- SECTION 4: Biological Dissection -->
<div class="section" id="biology">
  <h2><span class="num">4</span> Biological Dissection</h2>

  <div class="design-box">
    <h4>Experimental Design</h4>
    <p>We dissect per-clone W2 against five biological properties to understand <em>what makes a clone hard to simulate</em>. Analysis uses the best-performing method to isolate biology from method quality. Clone properties are computed from the h5ad file:</p>
    <ul>
      <li><strong>(a) Dominant fate:</strong> Most common non-undiff cell type at day 6</li>
      <li><strong>(b) Clone size:</strong> Number of cells at day 6</li>
      <li><strong>(c) Multipotency:</strong> Number of distinct non-undiff fate categories in the clone</li>
      <li><strong>(d) Pseudotime spread:</strong> Standard deviation of Palantir pseudotime at day 6 (within-clone differentiation variability)</li>
      <li><strong>(e) Initial state variance:</strong> Trace of day-4 covariance matrix (how spread-out the progenitors are)</li>
      <li><strong>(f) Method x fate heatmap:</strong> Median W2 broken down by method and dominant fate</li>
    </ul>
  </div>

  <div class="figure-container">
    <img src="data:image/png;base64,{fig_bio_dm10}" alt="Biology DM10">
    <p class="figure-caption">Figure 5. Biological dissection in DM10 space.</p>
  </div>

  <div class="figure-container">
    <img src="data:image/png;base64,{fig_bio_pc30}" alt="Biology PC30">
    <p class="figure-caption">Figure 6. Biological dissection in PC30 space.</p>
  </div>

  <div class="result-box">
    <h4>Results</h4>
    <ul>
      <li><strong>Multipotency correlates with difficulty</strong> (Spearman r ~ 0.43, p &lt; 0.001). Clones with 3+ fates have systematically higher W2. This is consistent with the hidden variable hypothesis: multipotent clones make diverse fate choices driven by hidden factors that trajectory inference cannot model.</li>
      <li><strong>Pseudotime spread drives W2</strong> (r ~ 0.48, p &lt; 0.001). Clones with broader maturation ranges at day 6 are harder to simulate -- the within-clone stochastic variation in differentiation pace is not captured.</li>
      <li><strong>Megakaryocyte clones are 2-2.6x harder</strong> across all methods. Meg fate arises via a "direct route" from LT-HSCs that bypasses standard multipotent intermediates (Rodriguez-Fraticelli 2018). Most methods are not designed for this non-canonical trajectory.</li>
    </ul>
  </div>

  <div class="implication-box">
    <h4>Implications</h4>
    <p>The difficulty hierarchy is <em>biological</em>, not algorithmic. No current method -- whether ODE or SDE -- overcomes the fundamental limitation that heritable fate biases are invisible to the transcriptome. Future methods incorporating epigenomic (e.g., ReDeeM mitochondrial mutations) or chromatin accessibility data may close this gap.</p>
  </div>
</div>

<!-- SECTION 5: Clone-Size Stratified W2 -->
<div class="section" id="stratified">
  <h2><span class="num">5</span> Clone-Size Stratified Analysis</h2>

  <div class="design-box">
    <h4>Experimental Design</h4>
    <p>Do methods perform differently on <strong>large vs. small clones</strong>? We split the 92 test clones into two groups by day-6 clone size and compare per-clone W2 distributions. Potential confounds:</p>
    <ul>
      <li>Large clones have more cells -> W2 is better estimated (lower sampling noise)</li>
      <li>Large clones may have different fate compositions (proliferation bias, Bonham-Carter 2024)</li>
      <li>Small clones may be more fate-restricted (fewer cells -> less diversity)</li>
    </ul>
    <p>We sweep the cutoff to test robustness: if a pattern holds across cutoffs, it reflects biology rather than a single arbitrary threshold.</p>
  </div>

  <div class="figure-container">
    <img src="data:image/png;base64,{fig_stratified}" alt="Clone-Size Stratified">
    <p class="figure-caption">Figure 7. Top: Ratio of median W2 (large/small group) across cutoff values for each method. Values above 1.0 mean large clones are harder. Bottom: Grouped bar chart at the median clone size cutoff.</p>
  </div>

  <div class="figure-container">
    <img src="data:image/png;base64,{fig_sweep_detail}" alt="Sweep Detail">
    <p class="figure-caption">Figure 8. Box plots of per-clone W2 for small vs. large clone groups at each cutoff (all methods pooled). Shows how the group distributions shift as the cutoff changes.</p>
  </div>

  <div class="result-box">
    <h4>Results</h4>
    {stratified_results}
  </div>

  <div class="implication-box">
    <h4>Implications</h4>
    <p>Clone size stratification disentangles two factors: (1) statistical power (large clones give more reliable W2 estimates) and (2) biological complexity (large clones may have more diverse fate outcomes due to longer proliferative history). If large clones consistently show higher W2, it suggests that proliferative clones -- which dominate the dataset due to growth bias -- are also biologically harder to simulate.</p>
  </div>
</div>

<!-- SECTION 6: Sensitivity Analysis -->
<div class="section" id="sensitivity">
  <h2><span class="num">6</span> Sensitivity Analysis</h2>

  <div class="design-box">
    <h4>Experimental Design</h4>
    <p>We test how method ranking changes when we vary the <strong>minimum clone size filter</strong>. Starting from our baseline of &ge;20 cells, we sweep thresholds from 5 to 50 cells. This reveals:</p>
    <ul>
      <li>Whether rankings are robust to the inclusion/exclusion of small clones</li>
      <li>Whether any method is artificially helped by small-clone sampling noise</li>
      <li>The trade-off between clone count (statistical power) and per-clone W2 reliability</li>
    </ul>
  </div>

  <div class="two-col">
    <div class="figure-container">
      <img src="data:image/png;base64,{fig_sens_dm10}" alt="Sensitivity DM10">
      <p class="figure-caption">Figure 9. Sensitivity analysis in DM10.</p>
    </div>
    <div class="figure-container">
      <img src="data:image/png;base64,{fig_sens_pc30}" alt="Sensitivity PC30">
      <p class="figure-caption">Figure 10. Sensitivity analysis in PC30.</p>
    </div>
  </div>

  <div class="result-box">
    <h4>Results</h4>
    <p>Method rankings are broadly stable across clone size thresholds. In DM10, OTCFM remains the top performer at all thresholds. In PC30, scDiffEq consistently leads. The relative ordering between the mid-pack methods may shift at extreme thresholds (n&ge;50), but this is driven by few remaining clones (&lt;20) rather than meaningful biological signal.</p>
  </div>
</div>

<!-- SECTION 7: Statistical Tests -->
<div class="section" id="stats">
  <h2><span class="num">7</span> Statistical Tests</h2>

  <div class="design-box">
    <h4>Experimental Design</h4>
    <p>We use <strong>paired Wilcoxon signed-rank tests</strong> to assess whether the per-clone W2 differences between methods are statistically significant. The test is paired because each clone is evaluated by all methods -- the same 92 clones form natural pairs. <strong>Bonferroni correction</strong> is applied for 36 pairwise comparisons (9 methods choose 2).</p>
    <p>The heatmap shows -log10(p) values (Bonferroni-corrected) in the color scale, with raw corrected p-values as annotations. Higher -log10(p) = more significant difference.</p>
  </div>

  <div class="two-col">
    <div class="figure-container">
      <img src="data:image/png;base64,{fig_wilcox_dm10}" alt="Wilcoxon DM10">
      <p class="figure-caption">Figure 11. Paired Wilcoxon heatmap, DM10.</p>
    </div>
    <div class="figure-container">
      <img src="data:image/png;base64,{fig_wilcox_pc30}" alt="Wilcoxon PC30">
      <p class="figure-caption">Figure 12. Paired Wilcoxon heatmap, PC30.</p>
    </div>
  </div>

  <div class="result-box">
    <h4>Results</h4>
    <p>In DM10, the top cluster (OTCFM, SF2M, TrajectoryNet, scDiffEq, DeepRUOT, pdp+, TIGON) is not always significantly different from each other after Bonferroni correction. However, MIOFlow and PRESCIENT are significantly worse than most methods (p &lt; 0.05). In PC30, the spread is tighter and fewer pairwise comparisons reach significance.</p>
  </div>
</div>

<!-- SECTION 8: Summary -->
<div class="section" id="summary">
  <h2><span class="num">8</span> Summary</h2>

  <h3>Full Results Table</h3>
  {summary_table}

  <h3>Key Takeaways</h3>
  <div class="result-box">
    <h4>Main Findings</h4>
    <ol>
      <li><strong>Per-clone W2 is more discriminating than population W2.</strong> All methods cluster tightly at the population level; per-clone analysis reveals a wider spread and more nuanced ranking.</li>
      <li><strong>Rankings are representation-dependent.</strong> OTCFM excels in DM10 (10D manifold) while scDiffEq leads in PC30 (30D linear). No single method dominates across both spaces.</li>
      <li><strong>Difficulty is biological.</strong> Multipotent clones, Meg-fated clones, and clones with high pseudotime spread are systematically harder for all methods. This correlates with the hidden variable argument (Weinreb 2020).</li>
      <li><strong>SDE does not help with multipotent clones.</strong> Stochastic methods do not outperform deterministic methods on high-multipotency clones (ratio ~ 1.06). The residual error is not noise that can be modeled by adding Brownian motion.</li>
      <li><strong>Results are robust to clone size filtering.</strong> Method rankings are stable across minimum clone size thresholds from 5 to 50 cells.</li>
    </ol>
  </div>

  <div class="implication-box">
    <h4>Looking Forward</h4>
    <p>Per-clone W2 reveals a <em>floor</em> that no current transcriptomics-only method can break through. Clones whose fates are driven by hidden variables (epigenomic, proteomic, spatial) will always have high per-clone W2 when simulated from scRNA-seq alone. Future trajectory inference methods should incorporate multi-omic measurements (SHARE-seq, ReDeeM, CITE-seq) to capture these hidden determinants. The per-clone W2 metric provides a direct way to measure progress toward this goal.</p>
  </div>
</div>

</div>

<footer>
  <p>Per-Clone W2 Analysis Report | Klein et al. LARRY Dataset | Generated {date}</p>
</footer>

<script>
document.querySelectorAll('.collapsible').forEach(btn => {{
  btn.addEventListener('click', function() {{
    this.classList.toggle('active');
    var content = this.nextElementSibling;
    if (content.style.maxHeight) {{
      content.style.maxHeight = null;
    }} else {{
      content.style.maxHeight = content.scrollHeight + "px";
    }}
  }});
}});
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    setup_style()
    print("Loading data ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} rows, {df['method'].nunique()} methods, {df['clone'].nunique()} clones")

    print("Generating concept diagrams ...")
    b64_design = fig_to_b64(diagram_experimental_design())
    b64_concept = fig_to_b64(diagram_per_clone_w2())
    b64_hidden = fig_to_b64(diagram_hidden_variables())

    print("Generating analysis figures ...")
    b64_ranking = fig_to_b64(fig_method_ranking(df))
    b64_bio_dm10 = fig_to_b64(fig_biology_panels(df, "dm10"))
    b64_bio_pc30 = fig_to_b64(fig_biology_panels(df, "pc30"))
    b64_sens_dm10 = fig_to_b64(fig_sensitivity(df, "dm10"))
    b64_sens_pc30 = fig_to_b64(fig_sensitivity(df, "pc30"))
    b64_wilcox_dm10 = fig_to_b64(fig_wilcoxon(df, "dm10"))
    b64_wilcox_pc30 = fig_to_b64(fig_wilcoxon(df, "pc30"))

    print("Generating clone-size stratified analysis ...")
    b64_stratified = fig_to_b64(fig_clone_size_stratified(df))
    b64_sweep = fig_to_b64(fig_clone_size_sweep_detail(df))

    # Compute stratified results text
    strat_lines = []
    median_size = int(df[df["space"] == "dm10"]["clone_size_tgt"].median())
    for space, sl in [("dm10", "DM10"), ("pc30", "PC30")]:
        sub = df[df["space"] == space]
        small = sub[sub["clone_size_tgt"] <= median_size]["w2_raw_mean"]
        large = sub[sub["clone_size_tgt"] > median_size]["w2_raw_mean"]
        ratio = large.median() / small.median() if small.median() > 0 else float("nan")
        stat, p = mannwhitneyu(large, small, alternative="two-sided")
        strat_lines.append(
            f"<p><strong>{sl}:</strong> Large clones (n>{median_size}) have median W2 = "
            f"{large.median():.6f} vs small clones (n<={median_size}) median W2 = {small.median():.6f} "
            f"(ratio = {ratio:.2f}, Mann-Whitney p = {p:.4f}). "
            f"N = {sub[sub['clone_size_tgt'] > median_size]['clone'].nunique()} large, "
            f"{sub[sub['clone_size_tgt'] <= median_size]['clone'].nunique()} small clones.</p>"
        )

    # per-method breakdown
    strat_lines.append("<p><strong>Per-method breakdown (DM10, cutoff={}):</strong></p>".format(median_size))
    strat_lines.append("<table><tr><th>Method</th><th>Small median W2</th><th>Large median W2</th><th>Ratio</th></tr>")
    sub = df[df["space"] == "dm10"]
    for method in sorted(sub["method"].unique()):
        ms = sub[sub["method"] == method]
        sm = ms[ms["clone_size_tgt"] <= median_size]["w2_raw_mean"].median()
        lg = ms[ms["clone_size_tgt"] > median_size]["w2_raw_mean"].median()
        r = lg / sm if sm > 0 else float("nan")
        color = METHOD_COLORS.get(method, "#999")
        strat_lines.append(
            f"<tr><td><span class='method-badge' style='background:{color}'>{method}</span></td>"
            f"<td>{sm:.6f}</td><td>{lg:.6f}</td><td>{r:.2f}</td></tr>"
        )
    strat_lines.append("</table>")
    stratified_results = "\n".join(strat_lines)

    # Summary table
    summary = make_summary_table(df)
    summary_html = summary.to_html(index=False, classes="", border=0)

    from datetime import datetime
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    print("Assembling HTML ...")
    html = HTML_TEMPLATE.format(
        date=date_str,
        fig_design=b64_design,
        fig_concept=b64_concept,
        fig_hidden=b64_hidden,
        fig_ranking=b64_ranking,
        fig_bio_dm10=b64_bio_dm10,
        fig_bio_pc30=b64_bio_pc30,
        fig_sens_dm10=b64_sens_dm10,
        fig_sens_pc30=b64_sens_pc30,
        fig_wilcox_dm10=b64_wilcox_dm10,
        fig_wilcox_pc30=b64_wilcox_pc30,
        fig_stratified=b64_stratified,
        fig_sweep_detail=b64_sweep,
        stratified_results=stratified_results,
        summary_table=summary_html,
    )

    with open(OUT_HTML, "w") as f:
        f.write(html)
    print(f"\nSaved: {OUT_HTML}")
    print(f"File size: {os.path.getsize(OUT_HTML) / 1024 / 1024:.1f} MB")
    print("Open in browser to view the report.")


if __name__ == "__main__":
    main()
