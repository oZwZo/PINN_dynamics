#!/usr/bin/env python
"""
04_make_ablation_configs.py
----------------------------
Generate all ablation-study JSON configs and a SLURM manifest for the
pseudodynamics+ weight-sensitivity sweep.

Usage
-----
  # Write files
  python scripts/pseudodynamics+/04_make_ablation_configs.py

  # Preview paths without writing
  python scripts/pseudodynamics+/04_make_ablation_configs.py --dry-run
"""

import argparse
import copy
import json
import os
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASELINE_KLEIN_PATH = os.path.join(
    REPO_ROOT, "logs", "klein_DM_10_lD1_lv1_lgNone", "V0_base.json"
)
LOGS_DIR = os.path.join(REPO_ROOT, "logs")
MANIFEST_PATH = os.path.join(LOGS_DIR, "ablation_manifest.txt")

# ---------------------------------------------------------------------------
# Per-stream baseline weights (defaults that apply to every arm in the stream).
# ---------------------------------------------------------------------------
# klein-NonOT and tom_pos: velocity supervised by deltax (RNA-velocity-style),
# CFM disabled because the OT assumption does not hold.
WEIGHT_BASELINE_NON_OT = {
    "D_penalty":        1,
    "deltax_weight":    1,
    "growth_weight":    1,
    "cfm_weight":       0,
    "neuralode_weight": 1,
    "R_weight":         1,
    "weight_intensity": 0,
    "D_var_weight":     0,
}

# klein-OT: velocity supervised by CFM only; deltax disabled (avoids
# duplicating the NonOT stream).
WEIGHT_BASELINE_OT = {
    "D_penalty":        1,
    "deltax_weight":    0,
    "growth_weight":    1,
    "cfm_weight":       1,
    "neuralode_weight": 1,
    "R_weight":         1,
    "weight_intensity": 0,
    "D_var_weight":     0,
}

# ---------------------------------------------------------------------------
# Sensitivity arms: (param_name, arm_label_prefix, list_of_non_default_values)
# ---------------------------------------------------------------------------
# klein-NonOT and tom_pos: do NOT sweep cfm_weight (held at 0)
NON_OT_ARMS = [
    ("D_penalty",        "lambdaD",         [0, 0.01, 10]),
    ("deltax_weight",    "lambdav",         [0, 0.01, 10]),
    ("growth_weight",    "lambdag",         [0, 0.01, 10]),
    ("neuralode_weight", "lambdaNeuralODE", [0, 0.01, 10]),
    ("R_weight",         "lambdaR",         [0, 100, 1000]),
]

# klein-OT: do NOT sweep deltax_weight (held at 0)
OT_ARMS = [
    ("D_penalty",        "lambdaD",         [0, 0.01, 10]),
    ("cfm_weight",       "lambdaCFM",       [0, 0.01, 10]),
    ("growth_weight",    "lambdag",         [0, 0.01, 10]),
    ("neuralode_weight", "lambdaNeuralODE", [0, 0.01, 10]),
    ("R_weight",         "lambdaR",         [0, 100, 1000]),
]

SYNTHETIC_ARMS = [
    ("D_penalty",        "lambdaD",         [0, 0.01, 10]),
]


def load_baseline_klein() -> dict:
    """Load raw_args from the Klein V0 baseline config."""
    with open(BASELINE_KLEIN_PATH, "r") as fh:
        cfg = json.load(fh)
    return dict(cfg["raw_args"])


def make_baseline_klein_pc(klein_raw: dict) -> dict:
    """Derive Klein PC-30 baseline (replaces the DM baseline)."""
    base = copy.deepcopy(klein_raw)
    base["dataset"]           = "klein_addpop"
    base["cellstate_key"]     = "X_pca_scaled"
    base["deltax_key"]        = "delta_PC"
    base["n_dimension"]       = 30
    base["timepoint_idx"]     = [0, 1, 2]
    base["batch_size"]        = 1024
    base["channels"]          = "256,256"
    base["norm_time"]         = "min_minus"
    base["time_sensitive"]    = True
    base["time_scale_factor"] = 1.0
    return base


def make_baseline_tompos(klein_raw: dict) -> dict:
    """Derive tom_pos baseline by overriding only dataset-specific keys."""
    base = copy.deepcopy(klein_raw)
    base["dataset"]       = "tom_pos"
    base["cellstate_key"] = "DM_scaled"
    base["deltax_key"]    = "Delta_DM"
    base["n_dimension"]   = 10
    base["timepoint_idx"] = [0, 1, 2, 3, 4, 6, 8]
    base["batch_size"]    = 50
    base["norm_time"]     = False
    base["channels"]      = "64,64"
    base["log_name"]      = "tom_pos_ablation/baseline"
    return base


def make_baseline_synthetic(klein_raw: dict) -> dict:
    """Derive synthetic_FP baseline by overriding dataset-specific keys."""
    base = copy.deepcopy(klein_raw)
    base["dataset"]       = "synthetic_FP"
    base["cellstate_key"] = "cellstate"
    base["n_dimension"]   = 2
    base["timepoint_idx"] = [0, 1, 2, 3, 4]
    base["deltax_key"]    = "delta_x"
    base["channels"]      = "256,256"
    base["log_name"]      = "synthetic_FP_ablation/baseline"
    return base


def _format_value(v) -> str:
    """Return a filesystem-safe string representation of a float/int."""
    if v == int(v):
        return str(int(v))
    # e.g. 0.01 -> '0.01'
    return str(v)


def build_configs(dataset_name: str, base_raw: dict, arms,
                  weight_baseline: dict) -> list[dict]:
    """
    Return a list of dicts:
        {"arm_name": str, "raw_args": dict}
    starting with the baseline, followed by one entry per (param, value) pair.
    """
    entries = []
    ablation_prefix = f"{dataset_name}_ablation"

    # --- Baseline ---
    baseline_raw = copy.deepcopy(base_raw)
    baseline_raw.update(weight_baseline)
    baseline_raw["config"]   = None
    baseline_raw["log_name"] = f"{ablation_prefix}/baseline"
    entries.append({"arm_name": "baseline", "raw_args": baseline_raw})

    # --- Sensitivity arms ---
    for param, label_prefix, values in arms:
        for val in values:
            arm_name = f"{label_prefix}_{_format_value(val)}"
            arm_raw  = copy.deepcopy(base_raw)
            arm_raw.update(weight_baseline)
            arm_raw[param]       = val
            arm_raw["config"]    = None
            arm_raw["log_name"]  = f"{ablation_prefix}/{arm_name}"
            entries.append({"arm_name": arm_name, "raw_args": arm_raw})

    return entries


def config_path(dataset_name: str, arm_name: str) -> str:
    """Absolute path where the config JSON will be written."""
    return os.path.join(
        LOGS_DIR,
        f"{dataset_name}_ablation",
        arm_name,
        "V0_config.json",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print would-be paths and manifest without writing any files.",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load baselines (klein DM config used only as a starting template)
    # -----------------------------------------------------------------------
    klein_dm_raw   = load_baseline_klein()
    klein_pc_raw   = make_baseline_klein_pc(klein_dm_raw)
    tompos_raw     = make_baseline_tompos(klein_dm_raw)
    synthetic_raw  = make_baseline_synthetic(klein_dm_raw)

    # Streams: (stream_name, base_raw, arms, weight_baseline)
    # klein-NonOT and klein-OT both use the PC-30 baseline.
    datasets = [
        ("klein_nonOT",  klein_pc_raw,  NON_OT_ARMS,    WEIGHT_BASELINE_NON_OT),
        ("klein_OT",     klein_pc_raw,  OT_ARMS,        WEIGHT_BASELINE_OT),
        ("tom_pos",      tompos_raw,    NON_OT_ARMS,    WEIGHT_BASELINE_NON_OT),
        ("synthetic_FP", synthetic_raw, SYNTHETIC_ARMS, WEIGHT_BASELINE_NON_OT),
    ]

    # -----------------------------------------------------------------------
    # Build all configs
    # -----------------------------------------------------------------------
    all_entries = []   # list of (dataset_name, arm_name, cfg_path, raw_args)

    for dataset_name, base_raw, arms, wbase in datasets:
        entries = build_configs(dataset_name, base_raw, arms, wbase)
        for e in entries:
            cfg_path_abs = config_path(dataset_name, e["arm_name"])
            all_entries.append(
                (dataset_name, e["arm_name"], cfg_path_abs, e["raw_args"])
            )

    total = len(all_entries)
    print(f"Total configs: {total}")
    print()

    # -----------------------------------------------------------------------
    # Dry-run: print paths and manifest preview
    # -----------------------------------------------------------------------
    if args.dry_run:
        print("=== Dry-run mode: no files will be written ===")
        print()
        for task_id, (dataset_name, arm_name, cfg_path_abs, _) in enumerate(all_entries):
            print(f"  [{task_id:02d}]  {cfg_path_abs}")
        print()
        print("=== Manifest preview (first 5 lines) ===")
        for task_id, (dataset_name, arm_name, cfg_path_abs, _) in enumerate(all_entries[:5]):
            print(f"  {task_id}\t{cfg_path_abs}\t{dataset_name}\t{arm_name}")
        if total > 5:
            print(f"  ... ({total - 5} more lines)")
        return

    # -----------------------------------------------------------------------
    # Write configs
    # -----------------------------------------------------------------------
    written = 0
    for task_id, (dataset_name, arm_name, cfg_path_abs, raw_args) in enumerate(all_entries):
        out_dir = os.path.dirname(cfg_path_abs)
        os.makedirs(out_dir, exist_ok=True)

        payload = {"raw_args": raw_args}
        with open(cfg_path_abs, "w") as fh:
            json.dump(payload, fh, indent=4)
        print(f"  [{task_id:02d}]  written  {cfg_path_abs}")
        written += 1

    # -----------------------------------------------------------------------
    # Write manifest
    # -----------------------------------------------------------------------
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w") as fh:
        for task_id, (dataset_name, arm_name, cfg_path_abs, _) in enumerate(all_entries):
            fh.write(f"{task_id}\t{cfg_path_abs}\t{dataset_name}\t{arm_name}\n")

    print()
    print(f"Wrote {written} config files.")
    print(f"Manifest written to: {MANIFEST_PATH}")
    print(f"  {total} lines  (SLURM array 0-{total - 1})")


if __name__ == "__main__":
    main()
