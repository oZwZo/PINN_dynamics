#!/usr/bin/env python
"""_generate_configs.py — Generate pdp config JSONs and SLURM manifests for cord blood.

Outputs
-------
- `logs/cordblood/pdp/<emb>_s<seed>/V0_config.json`   (per-arm pdp configs)
- `scripts/cordblood/manifest_tier1.tsv`              (TSV: task_id method embedding seed outdir)
- `scripts/cordblood/manifest_tier2.tsv`

Tiering (per the approved plan):
- Tier 1 (single SLURM array): pdp, prescient, otcfm, sf2m, deepruot (eval-only),
  scdiffeq (eval-only). 4 trainable + 2 eval-only methods.
- Tier 2 (separate array, launched after tier-1 passes): mioflow, tigon, trajectorynet.

PdP config notes
----------------
The CFM-aware trainer lives in the sibling `pseudodynamics_plus` repo:
`/rds/user/wz369/hpc-work/pseudodynamics_plus/main_train.py`. It exposes
`--cfm_weight`, `--growth_weight`, `--neuralode_weight`, `--R_weight`. The local
`PINN_dynamics/main_train.py` is the older non-CFM version. We use the
pseudodynamics_plus trainer (dispatched from `dispatch_task.sh`). The
pseudodynamics_plus repo's `data` directory is a symlink to PINN_dynamics/data,
so writing `data/cordblood_addpop.h5ad` makes it visible from both repos.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

ROOT = Path("/rds/user/wz369/hpc-work/PINN_dynamics")
LOGS_DIR = ROOT / "logs" / "cordblood"
SCRIPTS_DIR = ROOT / "scripts" / "cordblood"
ADATA_PATH = ROOT / "data" / "cordblood_addpop.h5ad"

SEEDS = [42, 7, 1234]

# Trainable + eval-only methods per tier
TIER1_TRAINABLE = [
    ("pdp",       ["PCA", "DM"]),
    ("prescient", ["PCA", "DM"]),
    ("otcfm",     ["PCA", "DM"]),
    ("sf2m",      ["PCA", "DM"]),
]
TIER1_EVAL_ONLY = [
    ("deepruot", ["PCA", "DM"]),
    ("scdiffeq", ["PCA", "DM"]),
]
TIER2_TRAINABLE = [
    ("mioflow",       ["PCA", "DM"]),
    ("tigon",         ["PCA", "DM"]),
    ("trajectorynet", ["PCA", "DM"]),
]

EMB_TO_KEY = {
    "PCA": "X_pca_scaled",
    "DM":  "DM_EigenVectors_scaled",
}

# OT baseline weights — cfm_weight=10 per the approved plan.
# Field set matches `pseudodynamics_plus/main_train.py`'s argparse (which exposes
# --cfm_weight, --growth_weight, --neuralode_weight, --R_weight).
PDP_BASELINE_RAW = {
    # required-by-trainer
    "config":             None,
    "dataset":            "cordblood_addpop",
    "cellstate_key":      None,   # set per-embedding
    "model":              "pde_params",
    "pretrained":         None,
    "gpu_devices":        0,
    "log_name":           None,   # set per-embedding+seed
    # learning
    "lr":                 3e-4,
    "schedule_lr":        "CyclicLR",
    "n_grid":             300,
    "n_dimension":        None,   # set per-embedding
    "timepoint_idx":      [0, 1, 2],
    "knn_volume":         "False",
    "batch_size":         512,
    "bw":                 None,
    "tol":                1e-4,
    "channels":           "256,256",
    "deltax_key":         None,   # set per-embedding
    "weight_intensity":   None,
    "time_scale_factor":  1.0,
    "norm_time":          "min_minus",
    "time_sensitive":     True,
    "progress_bar":       "False",
    # OT baseline weights (matches WEIGHT_BASELINE_OT in 04_make_ablation_configs.py)
    "D_penalty":          1.0,
    "deltax_weight":      0.0,    # OT: velocity supervised by CFM, not delta_x
    "growth_weight":      1.0,
    "cfm_weight":         10.0,   # per user request
    "neuralode_weight":   1.0,
    "R_weight":           1.0,
    "D_var_weight":       0.0,
}


def _read_dm_n_dims():
    """Read uns['dm_n_dims'] from the preprocessed cord blood h5ad."""
    if not ADATA_PATH.exists():
        raise FileNotFoundError(
            f"{ADATA_PATH} missing — run 01_preprocess.py first so we can read "
            f"uns['dm_n_dims']."
        )
    import scanpy as sc
    a = sc.read_h5ad(ADATA_PATH, backed="r")
    n = int(a.uns["dm_n_dims"])
    a.file.close()
    return n


def make_pdp_config(embedding: str, seed: int, dm_n_dims: int) -> dict:
    cfg = copy.deepcopy(PDP_BASELINE_RAW)
    cfg["seed"] = seed
    if embedding == "PCA":
        cfg["cellstate_key"] = "X_pca_scaled"
        cfg["deltax_key"]    = "delta_PC"
        cfg["n_dimension"]   = 30
        cfg["batch_size"]    = 1024
    elif embedding == "DM":
        cfg["cellstate_key"] = "DM_EigenVectors_scaled"
        cfg["deltax_key"]    = "delta_DM"
        # Use the actual palantir-determined DM dim count (after dropping the
        # trivial first eigenvector). Cord blood = 9.
        cfg["n_dimension"]   = dm_n_dims
        cfg["batch_size"]    = 512
    else:
        raise ValueError(f"Unknown embedding {embedding!r}")
    cfg["log_name"] = f"cordblood_pdp/{embedding}_s{seed}"
    return cfg


def outdir_for(method: str, embedding: str, seed: int) -> Path:
    emb_key = EMB_TO_KEY[embedding]
    return LOGS_DIR / method / f"{emb_key}_s{seed}"


def write_pdp_configs(seeds, dm_n_dims, dry_run=False):
    for embedding in ("PCA", "DM"):
        for seed in seeds:
            cfg = make_pdp_config(embedding, seed, dm_n_dims)
            out_dir = outdir_for("pdp", embedding, seed)
            cfg_path = out_dir / "V0_config.json"
            if not dry_run:
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(cfg_path, "w") as fh:
                    json.dump({"raw_args": cfg}, fh, indent=4)
            print(f"  pdp  {embedding}  seed={seed}  n_dim={cfg['n_dimension']} → {cfg_path}")


def write_manifest(path: Path, rows):
    with open(path, "w") as fh:
        fh.write("task_id\tmethod\tembedding\tseed\toutdir\n")
        for task_id, method, embedding, seed, outdir in rows:
            fh.write(f"{task_id}\t{method}\t{embedding}\t{seed}\t{outdir}\n")
    print(f"Wrote {path} ({len(rows)} rows)")


def build_manifest_rows(tier_methods, seeds, start_id=0):
    rows = []
    task_id = start_id
    for method, embeddings in tier_methods:
        for embedding in embeddings:
            emb_key = EMB_TO_KEY[embedding]
            for seed in seeds:
                outdir = outdir_for(method, embedding, seed)
                rows.append((task_id, method, emb_key, seed, str(outdir)))
                task_id += 1
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    print(f"Generating cord blood configs + manifests (dry_run={args.dry_run})")
    print(f"Seeds: {args.seeds}")
    dm_n_dims = _read_dm_n_dims()
    print(f"dm_n_dims (from uns): {dm_n_dims}\n")

    print("Writing pdp configs ...")
    write_pdp_configs(args.seeds, dm_n_dims, dry_run=args.dry_run)

    tier1_rows = build_manifest_rows(TIER1_TRAINABLE + TIER1_EVAL_ONLY, args.seeds, start_id=0)
    tier2_rows = build_manifest_rows(TIER2_TRAINABLE, args.seeds, start_id=0)
    print(f"\nTier 1: {len(tier1_rows)} tasks "
          f"({len(TIER1_TRAINABLE) + len(TIER1_EVAL_ONLY)} methods × 2 embeddings × {len(args.seeds)} seeds)")
    print(f"Tier 2: {len(tier2_rows)} tasks "
          f"({len(TIER2_TRAINABLE)} methods × 2 embeddings × {len(args.seeds)} seeds)")

    if not args.dry_run:
        SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        write_manifest(SCRIPTS_DIR / "manifest_tier1.tsv", tier1_rows)
        write_manifest(SCRIPTS_DIR / "manifest_tier2.tsv", tier2_rows)
        print(f"\nSLURM array ranges:")
        print(f"  tier 1: 0-{len(tier1_rows) - 1}  (update run_tier1.slurm --array=)")
        print(f"  tier 2: 0-{len(tier2_rows) - 1}  (update run_tier2.slurm --array=)")


if __name__ == "__main__":
    main()
