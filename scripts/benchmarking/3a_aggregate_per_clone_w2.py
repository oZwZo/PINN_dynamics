"""Aggregate per-clone W2 results from all trajectory inference methods.

Reads w2_per_clone_*.csv files from each method's eval output,
loads clone biological properties from the h5ad file, merges,
and exports a single aggregated DataFrame.

Usage:
    python 3a_aggregate_per_clone_w2.py
"""

import os
import re
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

HIGHER_DIR = "/rds/user/wz369/hpc-work"
PINN_DIR = os.path.join(HIGHER_DIR, "PINN_dynamics")
H5AD_PATH = os.path.join(PINN_DIR, "data/klein_addpop.h5ad")
CLONE_PROP_PATH = os.path.join(PINN_DIR, "data/klein/clone_proportions.csv")
OUT_PATH = os.path.join(PINN_DIR, "figures/klein_per_clone_w2_all.csv")
CLONE_PROP_OUT = os.path.join(PINN_DIR, "figures/klein_clone_properties.csv")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def _std_entries(base, runs, mode, fname_pattern="w2_per_clone_{mode}.csv"):
    """Build list of dicts for standard methods."""
    entries = []
    for label, subdir in runs:
        path = os.path.join(base, subdir, fname_pattern.format(mode=mode))
        entries.append({"path": path, "mode": mode, "label": label})
    return entries


def get_model_registry():
    """Return dict: method -> {dm10: [...], pc30: [...]}."""
    p = PINN_DIR
    h = HIGHER_DIR
    reg = {}

    # --- SF2M ---
    reg["SF2M"] = {
        "dm10": _std_entries(f"{p}/logs/sf2m/dm10_scaled",
                            [("r1","model_r1"),("r2","model_r2"),("r3","model_r3")], "sde"),
        "pc30": _std_entries(f"{p}/logs/sf2m/pca30",
                            [("r1","model_r1"),("r2","model_r2"),("r3","model_r3")], "sde"),
    }

    # --- OTCFM ---
    reg["OTCFM"] = {
        "dm10": _std_entries(f"{p}/logs/otcfm/dm10",
                            [("r1","model_r1"),("r2","model_r2"),("r3","model_r3")], "ode"),
        "pc30": _std_entries(f"{p}/logs/otcfm/pca30",
                            [("r1","model_r1"),("r2","model_r2"),("r3","model_r3")], "ode"),
    }

    # --- TrajectoryNet ---
    reg["TrajectoryNet"] = {
        "dm10": _std_entries(f"{p}/logs/TrajectoryNet/dm10",
                            [("r1","model"),("r2","model_run2"),("r3","model_run3")], "ode"),
        "pc30": _std_entries(f"{p}/logs/TrajectoryNet/pca30",
                            [("r1","model"),("r2","model_run2"),("r3","model_run3"),("r4","model_run4")], "ode"),
    }

    # --- TIGON ---
    reg["TIGON"] = {
        "dm10": _std_entries(f"{p}/logs/TIGON/dm",
                            [("s1","model"),("s2","model_seed2"),("s3","model_seed3")], "ode"),
        "pc30": _std_entries(f"{p}/logs/TIGON/pca",
                            [("s1","model"),("s2","model_seed2"),("s3","model_seed3")], "ode"),
    }

    # --- MIOFlow ---
    reg["MIOFlow"] = {
        "dm10": _std_entries(f"{p}/logs/MIOFlow",
                            [("r1","klein_dm_gaga_run1"),("r2","klein_dm_gaga_run2"),("r3","klein_dm_gaga_run3")], "ode"),
        "pc30": _std_entries(f"{p}/logs/MIOFlow",
                            [("r1","klein_pca_gaga_run1"),("r2","klein_pca_gaga_run2"),("r3","klein_pca_gaga_run3")], "ode"),
    }

    # --- DeepRUOT (external) ---
    reg["DeepRUOT"] = {
        "dm10": _std_entries(f"{h}/DeepRUOTv2/results",
                            [("s0","klein_dm10"),("s1","klein_dm10_s0"),("s2","klein_dm10_s10"),("s3","klein_dm10_s42")], "ode"),
        "pc30": _std_entries(f"{h}/DeepRUOTv2/results",
                            [("s0","klein_pca30"),("s1","klein_pca30_s0"),("s2","klein_pca30_s10"),("s3","klein_pca30_s42")], "ode"),
    }

    # --- PRESCIENT ---
    reg["PRESCIENT"] = {
        "dm10": _std_entries(f"{p}/results/PRESCIENT/dm_scaled_run/DM_SCALED-softplus_4_64-1e-06",
                            [("s0","seed_0"),("s2","seed_2"),("s42","seed_42")], "sde"),
        "pc30": _std_entries(f"{p}/results/PRESCIENT/pca_scaled_run/PCA_SCALED-softplus_1_500-1e-06",
                            [("s1","seed_1"),("s11","seed_11"),("s22","seed_22")], "sde"),
    }

    # --- scDiffEq (external) ---
    reg["scDiffEq"] = {
        "dm10": [{"path": f"{h}/scDiffEq/results/seeds/seed_{s}/dm10/plain_sde/fate_prediction_metrics/last/w2_per_clone_sde.csv",
                  "mode": "sde", "label": f"s{s}"} for s in [0,1,2]],
        "pc30": [{"path": f"{h}/scDiffEq/results/seeds/seed_{s}/pca30/plain_sde/fate_prediction_metrics/last/w2_per_clone_sde.csv",
                  "mode": "sde", "label": f"s{s}"} for s in [0,1,2]],
    }

    # --- pdp+ (special handling: select best config) ---
    reg["pdp+"] = {
        "dm10": _get_pdp_best_per_clone("dm10"),
        "pc30": _get_pdp_best_per_clone("pc30"),
    }

    return reg


def _get_pdp_best_per_clone(space):
    """For pdp+, scan all configs & hyperparams, select best by pop w2_raw."""
    pdp_base = f"{HIGHER_DIR}/pseudodynamics_plus/results/pseudodynamics+/"
    if space == "dm10":
        config_patterns = ["klein_DMscaled_10_cfm5_b512", "klein_DMscaled_10_cfm5_b1024",
                           "klein_DMscaled_10_cfm10_b512", "klein_DMscaled_10_cfm10_b1024"]
    else:
        config_patterns = ["klein_PC_30_lD1_lv1_lgNone",
                           "klein_PC30_lD1_cfm1_lgNone", "klein_PC30_lD1_cfm1_lv1_lgNone",
                           "klein_PC30_lD1_cfm2_lgNone", "klein_PC30_lD1_cfm10_lgNone",
                           "klein_PC30_lD1_cfm10_lgNone_b1024"]

    best_w2 = float("inf")
    best_per_clone_path = None

    for config in config_patterns:
        config_dir = os.path.join(pdp_base, config)
        if not os.path.isdir(config_dir):
            continue
        for fname in os.listdir(config_dir):
            if fname.endswith("_w2_eval.csv"):
                pop_path = os.path.join(config_dir, fname)
                try:
                    pop_df = pd.read_csv(pop_path)
                    w2 = pop_df["w2_raw"].iloc[0]
                except Exception:
                    continue
                per_clone_fname = fname.replace("_w2_eval.csv", "_w2_eval_per_clone.csv")
                per_clone_path = os.path.join(config_dir, per_clone_fname)
                if w2 < best_w2 and os.path.exists(per_clone_path):
                    best_w2 = w2
                    best_per_clone_path = per_clone_path

    if best_per_clone_path is None:
        log.warning(f"pdp+ {space}: no per-clone W2 file found")
        return []

    m = re.search(r"(t[\d.]+_n[\d.]+_\w+)_w2_eval_per_clone\.csv", os.path.basename(best_per_clone_path))
    mode_str = m.group(1).split("_")[-1] if m else "ode"
    log.info(f"pdp+ {space}: best config = {best_per_clone_path} (pop w2_raw={best_w2:.6f})")

    return [{"path": best_per_clone_path, "mode": mode_str, "label": "best"}]


# ---------------------------------------------------------------------------
# Load per-clone W2
# ---------------------------------------------------------------------------

def load_per_clone_w2(registry):
    """Read all per-clone CSVs and concatenate."""
    rows = []
    for method, spaces in registry.items():
        for space, entries in spaces.items():
            for entry in entries:
                path = entry["path"]
                if not os.path.exists(path):
                    log.warning(f"MISSING: {method}/{space}/{entry['label']}: {path}")
                    continue
                df = pd.read_csv(path)
                # Normalise column names (pdp+ uses clone_size_t4/t6)
                df = df.rename(columns={"clone_size_t4": "clone_size_src",
                                        "clone_size_t6": "clone_size_tgt"})
                df["method"] = method
                df["space"] = space
                df["run"] = entry["label"]
                df["mode"] = entry["mode"]
                rows.append(df)
    if not rows:
        raise RuntimeError("No per-clone W2 files found!")
    all_df = pd.concat(rows, ignore_index=True)
    log.info(f"Loaded {len(all_df)} rows from {len(rows)} files, "
             f"{all_df['method'].nunique()} methods, "
             f"{all_df['clone'].nunique()} unique clones")
    return all_df


def average_across_runs(df):
    """Average per-clone W2 across seeds/runs for each method × space × clone."""
    agg = (df.groupby(["method", "space", "clone"])
           .agg(w2_raw_mean=("w2_raw", "mean"),
                w2_raw_std=("w2_raw", "std"),
                w2_scaled_mean=("w2_scaled", "mean"),
                n_runs=("w2_raw", "count"),
                clone_size_src=("clone_size_src", "first"),
                clone_size_tgt=("clone_size_tgt", "first"))
           .reset_index())
    agg["w2_raw_std"] = agg["w2_raw_std"].fillna(0)
    return agg


# ---------------------------------------------------------------------------
# Clone biological properties
# ---------------------------------------------------------------------------

def load_clone_properties(h5ad_path, clone_ids):
    """Compute clone properties from the h5ad file."""
    import anndata as ad

    log.info(f"Loading {h5ad_path} ...")
    adata = ad.read_h5ad(h5ad_path)
    test = adata[adata.obs["Well"] == 2].copy()
    test = test[test.obs["clones"].isin(clone_ids)]

    d6 = test[test.obs["Time_point"] == "6.0"].copy()
    d4 = test[test.obs["Time_point"] == "4.0"].copy()

    records = []
    for cid in clone_ids:
        rec = {"clone": cid}

        # --- Day 6 properties ---
        mask6 = d6.obs["clones"] == cid
        cells6 = d6[mask6]
        rec["clone_size_d6"] = int(mask6.sum())

        if rec["clone_size_d6"] > 0:
            ann6 = cells6.obs["Annotation"]
            non_undiff = ann6[ann6 != "undiff"]
            if len(non_undiff) > 0:
                rec["dominant_fate"] = non_undiff.value_counts().idxmax()
                rec["multipotency"] = non_undiff.nunique()
            else:
                rec["dominant_fate"] = "undiff"
                rec["multipotency"] = 0
            pt = cells6.obs["palantir_pseudotime"].dropna()
            rec["pseudotime_spread"] = float(pt.std()) if len(pt) > 1 else 0.0
        else:
            rec["dominant_fate"] = np.nan
            rec["multipotency"] = np.nan
            rec["pseudotime_spread"] = np.nan

        # --- Day 4 properties ---
        mask4 = d4.obs["clones"] == cid
        cells4 = d4[mask4]
        rec["clone_size_d4"] = int(mask4.sum())

        if rec["clone_size_d4"] > 1:
            dm_emb = cells4.obsm["DM_EigenVectors_scaled"]
            rec["initial_var_dm"] = float(np.trace(np.cov(dm_emb.T)))
            pc_emb = cells4.obsm["X_pca_scaled"][:, :30]
            rec["initial_var_pc"] = float(np.trace(np.cov(pc_emb.T)))
        else:
            rec["initial_var_dm"] = np.nan
            rec["initial_var_pc"] = np.nan

        records.append(rec)

    props = pd.DataFrame(records)
    log.info(f"Computed properties for {len(props)} clones, "
             f"fates: {props['dominant_fate'].value_counts().to_dict()}")
    return props


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    registry = get_model_registry()
    df_raw = load_per_clone_w2(registry)
    df_mean = average_across_runs(df_raw)

    clone_ids = sorted(df_mean["clone"].unique())

    clone_props = load_clone_properties(H5AD_PATH, clone_ids)
    clone_props.to_csv(CLONE_PROP_OUT, index=False)
    log.info(f"Saved clone properties to {CLONE_PROP_OUT}")

    df_out = df_mean.merge(clone_props, on="clone", how="left")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df_out.to_csv(OUT_PATH, index=False)
    log.info(f"Saved aggregated data ({len(df_out)} rows) to {OUT_PATH}")

    print("\n=== Summary ===")
    summary = (df_out.groupby(["method", "space"])
               .agg(median_w2=("w2_raw_mean", "median"),
                    mean_w2=("w2_raw_mean", "mean"),
                    n_clones=("clone", "nunique"),
                    n_runs_per_clone=("n_runs", "first"))
               .reset_index()
               .sort_values(["space", "median_w2"]))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
