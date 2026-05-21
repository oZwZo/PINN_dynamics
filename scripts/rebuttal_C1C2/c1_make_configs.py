#!/usr/bin/env python
"""
Emit 5 JSON configs for the C1 λ_growth sweep on the 5-D synthetic FP dataset
(syn_cell_0_4.h5ad — the manuscript Sup Fig 2 dataset), and an array_index.tsv
that maps SLURM array task → seed → 5 config paths.

Hyperparameters match the Sup Fig 2 setup (per
/rds/user/wz369/hpc-work/PINN_dynamics/Explore_Notebook/11.sythsize_data/2.train_model.py)
as closely as main_train.py allows. Two unavoidable deviations:
  - main_train.py shares one --channels for g/v/D networks (no asymmetric widths).
  - main_train.py does random 80:10:10 cell-level split inside --timepoint_idx,
    not the deterministic [0,1,2,4,6,8,10]/[5,7]/[3,9] timepoint hold-out.
Both are disclosed in the plan and the supplementary methods.

Compute layout: 3 SLURM tasks (one per seed) × 5 parallel trainings on a single
A100 per task. The TSV is therefore one row per seed, with 5 config paths.
"""
import argparse
import json
from pathlib import Path


# user choices fixed in this version (see .claude/plans/serene-dreaming-jellyfish.md):
DATASET = "synthetic_FP_5D"
CELLSTATE_KEY = "X_data"
DELTAX_KEY = "Delta_DM"
N_DIM = 5
TIMEPOINT_IDX = list(range(11))
LAMBDA_GROWTH_VALUES = [0, 1, 3, 10, 100]
SEEDS = [0, 1, 2]
LOG_ROOT = "logs/synthetic_FP_5D_ablation"

# Sup Fig 2 hyperparameters (per 2.train_model.py)
BASE_CONFIG = {
    "config": None,
    "dataset": DATASET,
    "cellstate_key": CELLSTATE_KEY,
    "model": "pde_params",
    "pretrained": None,
    "gpu_devices": 0,
    "log_name": None,            # filled per-arm per-seed by SLURM
    "lr": 0.0003,
    "schedule_lr": "CyclicLR",
    "n_grid": 300,
    "n_dimension": N_DIM,
    "timepoint_idx": TIMEPOINT_IDX,
    "knn_volume": "False",
    "batch_size": 200,
    "bw": None,
    "tol": 0.0001,
    "channels": "16,16",         # gives g=[6,16,16,1]; v=[6,16,16,5]; D=[6,16,1]
    "D_penalty": 0.1,            # matches Sup Fig 2
    "deltax_key": DELTAX_KEY,
    "deltax_weight": 1,
    "weight_intensity": 3,       # matches Sup Fig 2
    "time_scale_factor": 1.0,
    "norm_time": "min_minus",
    "time_sensitive": True,
    "progress_bar": "False",
    "growth_weight": None,       # filled per-arm
    "R_weight": 1,
    "cfm_weight": 0,
    "neuralode_weight": 1,
    "D_var_weight": 0,
}


def arm_dir(lg: float) -> str:
    """Stable on-disk arm name. Avoid '.' in directory names for regex friendliness."""
    return f"lambdag_{lg}".replace(".", "p")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default="/rds/user/wz369/hpc-work/pseudodynamics_plus",
    )
    args = parser.parse_args()
    repo_root = Path(args.repo_root)

    # 1. Emit per-arm configs
    arm_to_cfg: dict[float, Path] = {}
    for lg in LAMBDA_GROWTH_VALUES:
        arm = arm_dir(lg)
        arm_path = repo_root / LOG_ROOT / arm
        arm_path.mkdir(parents=True, exist_ok=True)

        cfg = dict(BASE_CONFIG)
        cfg["log_name"] = f"{LOG_ROOT}/{arm}"
        cfg["growth_weight"] = lg
        cfg_path = arm_path / "V0_config.json"
        cfg_path.write_text(json.dumps({"raw_args": cfg}, indent=4))
        arm_to_cfg[lg] = cfg_path

    # 2. Emit array_index.tsv (one row per seed; 5 config paths per row)
    rows = []
    for seed in SEEDS:
        cfg_paths = [str(arm_to_cfg[lg]) for lg in LAMBDA_GROWTH_VALUES]
        rows.append((seed, cfg_paths))

    index_path = repo_root / LOG_ROOT / "array_index.tsv"
    header = ["seed"] + [f"cfg_{lg}" for lg in LAMBDA_GROWTH_VALUES]
    with open(index_path, "w") as f:
        f.write("\t".join(header) + "\n")
        for seed, paths in rows:
            f.write("\t".join([str(seed), *paths]) + "\n")

    # 3. SLURM dirs
    (repo_root / LOG_ROOT / "slurm").mkdir(parents=True, exist_ok=True)

    # 4. Prepared dataset.
    # The reader needs additional fields (timepoint_tx_days, pop['std'/'var'/
    # 'n_lib']) that the raw synthesis h5ad does not carry. Build/refresh the
    # prepared file once here (NOT per SLURM task — avoids race).
    dst_h5ad = repo_root / "data" / f"{DATASET}.h5ad"
    if dst_h5ad.exists():
        print(f"prepared dataset already at {dst_h5ad} ({dst_h5ad.stat().st_size/1e6:.1f} MB) — skipping rebuild")
    else:
        from importlib import import_module
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        c1_prep = import_module("c1_prepare_h5ad")
        c1_prep.main()

    print(f"wrote {len(LAMBDA_GROWTH_VALUES)} configs under {repo_root / LOG_ROOT}/lambdag_*/V0_config.json")
    print(f"wrote {len(rows)} array rows  →  {index_path}")
    print(f"sweep grid: λ_growth ∈ {LAMBDA_GROWTH_VALUES}, seeds = {SEEDS}")


if __name__ == "__main__":
    main()
