"""Plot per-clone W2 analysis: method ranking, CV, biology, sensitivity, stats.

Reads the aggregated CSV from 3a_aggregate_per_clone_w2.py and generates
publication-ready figures.

Usage:
    python 3b_plot_per_clone_w2.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, spearmanr
from itertools import combinations

PINN_DIR = "/rds/user/wz369/hpc-work/PINN_dynamics"
DATA_PATH = os.path.join(PINN_DIR, "figures/klein_per_clone_w2_all.csv")
FIG_DIR = os.path.join(PINN_DIR, "figures")

METHOD_ORDER = ["pdp+", "DeepRUOT", "scDiffEq", "OTCFM", "SF2M",
                "TrajectoryNet", "TIGON", "PRESCIENT", "MIOFlow"]

METHOD_COLORS = {
    'OTCFM': '#505175',
    'SF2M': '#7C8DBE',
    'pdp+': '#C5D9ED',
    'PRESCIENT': '#FFE7BD',
    'TrajectoryNet': '#EFB397',
    'MIOFlow': '#D087A3',
    'TIGON': '#A1689E',
    'scDiffEq': '#88B5A3',
    'DeepRUOT': '#D4856E'
}

# ---------------------------------------------------------------------------
# Publication styling
# ---------------------------------------------------------------------------

def setup_style():
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    sns.set_style("ticks")


# ---------------------------------------------------------------------------
# Figure 1: Method ranking violin plots
# ---------------------------------------------------------------------------

def plot_method_ranking(df, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, space, title in zip(axes, ["dm10", "pc30"], ["DM10", "PC30"]):
        sub = df[df["space"] == space].copy()
        if sub.empty:
            ax.set_title(f"{title} (no data)")
            continue

        order = (sub.groupby("method")["w2_raw_mean"]
                 .median().sort_values().index.tolist())

        palette = {m: METHOD_COLORS.get(m, "#999") for m in order}

        sns.violinplot(data=sub, x="method", y="w2_raw_mean", order=order,
                       palette=palette, inner=None, alpha=0.3, ax=ax, cut=0)
        sns.stripplot(data=sub, x="method", y="w2_raw_mean", order=order,
                      palette=palette, size=3, alpha=0.5, jitter=True, ax=ax)
        sns.boxplot(data=sub, x="method", y="w2_raw_mean", order=order,
                    color="white", width=0.3, fliersize=0, ax=ax,
                    boxprops=dict(alpha=0.7), medianprops=dict(color="black", linewidth=2))

        ax.set_xlabel("")
        ax.set_ylabel("Per-clone W2 (raw)")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "per_clone_w2_method_ranking.pdf")
        fig.savefig(path)
        print(f"Saved: {path}")
    return fig


# ---------------------------------------------------------------------------
# Figure 2: CV analysis
# ---------------------------------------------------------------------------

def plot_cv_analysis(df, save=True):
    records = []
    for (method, space), grp in df.groupby(["method", "space"]):
        vals = grp["w2_raw_mean"].values
        cv = vals.std() / vals.mean() if vals.mean() > 0 else np.nan
        records.append({"method": method, "space": space, "CV": cv})
    cv_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(10, 5))
    dm_order = (cv_df[cv_df["space"] == "dm10"]
                .sort_values("CV")["method"].tolist())
    order = dm_order + [m for m in cv_df["method"].unique() if m not in dm_order]

    sns.barplot(data=cv_df, x="method", y="CV", hue="space",
                order=order, palette={"dm10": "#457B9D", "pc30": "#E9C46A"},
                ax=ax, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("")
    ax.set_ylabel("Coefficient of Variation\n(per-clone W2)")
    ax.set_title("Per-clone W2 variability across clones")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Space")
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "per_clone_w2_cv_analysis.pdf")
        fig.savefig(path)
        print(f"Saved: {path}")
    return fig


# ---------------------------------------------------------------------------
# Figure 3: Biological dissection
# ---------------------------------------------------------------------------

def plot_biology_panels(df, space="dm10", save=True):
    sub = df[df["space"] == space].copy()
    if sub.empty:
        print(f"No data for space={space}")
        return None

    methods_present = sub["method"].unique()
    best_method = sub.groupby("method")["w2_raw_mean"].median().idxmin()
    best = sub[sub["method"] == best_method].copy()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    space_label = "DM10" if space == "dm10" else "PC30"

    # (a) W2 by dominant fate
    ax = axes[0, 0]
    fate_counts = best["dominant_fate"].value_counts()
    fates_to_show = fate_counts[fate_counts >= 3].index.tolist()
    d = best[best["dominant_fate"].isin(fates_to_show)]
    if not d.empty:
        fate_order = (d.groupby("dominant_fate")["w2_raw_mean"]
                      .median().sort_values().index.tolist())
        sns.boxplot(data=d, x="dominant_fate", y="w2_raw_mean",
                    order=fate_order, ax=ax, palette="Set2")
        sns.stripplot(data=d, x="dominant_fate", y="w2_raw_mean",
                      order=fate_order, ax=ax, color="black", size=3, alpha=0.5)
    ax.set_xlabel("Dominant fate (day 6)")
    ax.set_ylabel("Per-clone W2")
    ax.set_title(f"(a) W2 by fate type ({best_method})")
    ax.tick_params(axis="x", rotation=45)

    # (b) W2 vs clone size (day 6)
    ax = axes[0, 1]
    _scatter_with_corr(best, "clone_size_tgt", "w2_raw_mean", ax,
                       "Clone size (day 6)", "Per-clone W2",
                       f"(b) W2 vs clone size ({best_method})")

    # (c) W2 vs multipotency
    ax = axes[0, 2]
    multi_col = "multipotency"
    if multi_col in best.columns and best[multi_col].notna().sum() > 5:
        d = best.dropna(subset=[multi_col])
        d[multi_col] = d[multi_col].astype(int)
        multi_vals = sorted(d[multi_col].unique())
        sns.boxplot(data=d, x=multi_col, y="w2_raw_mean",
                    order=multi_vals, ax=ax, palette="Blues")
        sns.stripplot(data=d, x=multi_col, y="w2_raw_mean",
                      order=multi_vals, ax=ax, color="black", size=3, alpha=0.5)
        r, p = spearmanr(d[multi_col], d["w2_raw_mean"])
        ax.set_title(f"(c) W2 vs multipotency (r={r:.2f}, p={p:.3f})")
    else:
        ax.set_title("(c) W2 vs multipotency (insufficient data)")
    ax.set_xlabel("Multipotency (n fates)")
    ax.set_ylabel("Per-clone W2")

    # (d) W2 vs pseudotime spread
    ax = axes[1, 0]
    _scatter_with_corr(best, "pseudotime_spread", "w2_raw_mean", ax,
                       "Pseudotime spread (std, day 6)", "Per-clone W2",
                       f"(d) W2 vs pseudotime spread ({best_method})")

    # (e) W2 vs initial state variance
    ax = axes[1, 1]
    var_col = "initial_var_dm" if space == "dm10" else "initial_var_pc"
    _scatter_with_corr(best, var_col, "w2_raw_mean", ax,
                       "Initial state variance (day 4)", "Per-clone W2",
                       f"(e) W2 vs initial variance ({best_method})")

    # (f) W2 by fate — all methods heatmap
    ax = axes[1, 2]
    if "dominant_fate" in sub.columns:
        pivot = (sub.groupby(["method", "dominant_fate"])["w2_raw_mean"]
                 .median().reset_index()
                 .pivot(index="method", columns="dominant_fate", values="w2_raw_mean"))
        pivot = pivot.reindex(columns=fates_to_show) if fates_to_show else pivot
        if not pivot.empty:
            sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlOrRd", ax=ax,
                        linewidths=0.5, cbar_kws={"label": "Median W2"})
            ax.set_title("(f) Median W2 by method × fate")
            ax.set_ylabel("")
        else:
            ax.set_title("(f) No data")
    else:
        ax.set_title("(f) No fate data")

    fig.suptitle(f"Biological dissection — {space_label}", fontsize=16, y=1.02)
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, f"per_clone_w2_biology_{space}.pdf")
        fig.savefig(path)
        print(f"Saved: {path}")
    return fig


def _scatter_with_corr(data, xcol, ycol, ax, xlabel, ylabel, title):
    d = data.dropna(subset=[xcol, ycol])
    if len(d) < 5:
        ax.set_title(f"{title} (insufficient data)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return
    ax.scatter(d[xcol], d[ycol], alpha=0.6, edgecolors="black", linewidth=0.3, s=30)
    r, p = spearmanr(d[xcol], d[ycol])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n(Spearman r={r:.2f}, p={p:.3f})")


# ---------------------------------------------------------------------------
# Figure 4: Sensitivity analysis
# ---------------------------------------------------------------------------

def plot_sensitivity(df, space="dm10", save=True):
    thresholds = [5, 10, 20, 30, 50]
    records = []
    for thresh in thresholds:
        sub = df[(df["space"] == space) & (df["clone_size_tgt"] >= thresh)]
        for method, grp in sub.groupby("method"):
            records.append({
                "threshold": thresh,
                "method": method,
                "median_w2": grp["w2_raw_mean"].median(),
                "n_clones": grp["clone"].nunique(),
            })
    sens_df = pd.DataFrame(records)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    space_label = "DM10" if space == "dm10" else "PC30"

    for method in sens_df["method"].unique():
        d = sens_df[sens_df["method"] == method]
        color = METHOD_COLORS.get(method, "#999")
        ax1.plot(d["threshold"], d["median_w2"], "o-", label=method,
                 color=color, markersize=5)
        ax2.plot(d["threshold"], d["n_clones"], "o-", label=method,
                 color=color, markersize=5)

    ax1.set_xlabel("Min clone size (day 6)")
    ax1.set_ylabel("Median per-clone W2")
    ax1.set_title(f"W2 sensitivity to clone size filter ({space_label})")
    ax1.legend(fontsize=8, ncol=2)

    ax2.set_xlabel("Min clone size (day 6)")
    ax2.set_ylabel("Number of clones")
    ax2.set_title("Clone count at each threshold")

    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, f"per_clone_w2_sensitivity_{space}.pdf")
        fig.savefig(path)
        print(f"Saved: {path}")
    return fig


# ---------------------------------------------------------------------------
# Figure 5: Wilcoxon heatmap
# ---------------------------------------------------------------------------

def plot_wilcoxon_heatmap(df, space="dm10", save=True):
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
            stat, p = wilcoxon(shared[m1], shared[m2])
        except ValueError:
            p = 1.0
        pval_matrix.loc[m1, m2] = p
        pval_matrix.loc[m2, m1] = p

    n_tests = n * (n - 1) / 2
    pval_bonf = pval_matrix.clip(upper=1.0) * n_tests
    pval_bonf = pval_bonf.clip(upper=1.0)
    np.fill_diagonal(pval_bonf.values, np.nan)

    log_p = -np.log10(pval_bonf.replace(0, 1e-300))
    np.fill_diagonal(log_p.values, 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    space_label = "DM10" if space == "dm10" else "PC30"
    mask = np.triu(np.ones_like(log_p, dtype=bool), k=0)

    annot = pval_bonf.copy()
    annot_str = annot.map(lambda x: f"{x:.3f}" if pd.notna(x) else "")

    sns.heatmap(log_p, mask=mask, annot=annot_str, fmt="",
                cmap="YlOrRd", ax=ax, linewidths=0.5,
                cbar_kws={"label": "-log10(p, Bonferroni)"}, vmin=0, vmax=5)
    ax.set_title(f"Paired Wilcoxon signed-rank test ({space_label})")
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, f"per_clone_w2_wilcoxon_{space}.pdf")
        fig.savefig(path)
        print(f"Saved: {path}")
    return fig


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(df):
    records = []
    for (method, space), grp in df.groupby(["method", "space"]):
        vals = grp["w2_raw_mean"].values
        records.append({
            "method": method,
            "space": space,
            "median_w2": np.median(vals),
            "mean_w2": np.mean(vals),
            "std_w2": np.std(vals),
            "CV": np.std(vals) / np.mean(vals) if np.mean(vals) > 0 else np.nan,
            "n_clones": len(vals),
            "n_runs": int(grp["n_runs"].iloc[0]),
        })
    summary = pd.DataFrame(records).sort_values(["space", "median_w2"])
    print("\n=== Per-Clone W2 Summary ===")
    print(summary.to_string(index=False))

    path = os.path.join(FIG_DIR, "klein_per_clone_w2_summary.csv")
    summary.to_csv(path, index=False)
    print(f"\nSaved: {path}")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_style()
    os.makedirs(FIG_DIR, exist_ok=True)

    print(f"Reading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows, {df['method'].nunique()} methods, "
          f"{df['clone'].nunique()} clones")

    plot_method_ranking(df)
    plot_cv_analysis(df)

    for space in ["dm10", "pc30"]:
        plot_biology_panels(df, space=space)
        plot_sensitivity(df, space=space)
        plot_wilcoxon_heatmap(df, space=space)

    print_summary_table(df)
    print("\nDone!")


if __name__ == "__main__":
    main()
