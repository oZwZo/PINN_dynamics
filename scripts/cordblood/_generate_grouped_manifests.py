#!/usr/bin/env python
"""_generate_grouped_manifests.py — Generate manifests where one task = one
(method, embedding) cell. The SLURM script iterates the 3 seeds internally on
the same GPU (with random sleep), then calls eval on each best checkpoint.

Outputs:
  scripts/cordblood/manifest_tier1_grouped.tsv  (8 rows: 4 trainable methods × 2 embeddings)
  scripts/cordblood/manifest_tier2_grouped.tsv  (6 rows: 3 methods × 2 embeddings)
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path("/rds/user/wz369/hpc-work/PINN_dynamics")
SCRIPTS_DIR = ROOT / "scripts" / "cordblood"
LOGS_DIR = ROOT / "logs" / "cordblood"

# Trainable tier-1 methods (eval-only deepruot / scdiffeq are handled separately).
TIER1_TRAINABLE = ["pdp", "prescient", "otcfm", "sf2m"]
TIER2_TRAINABLE = ["mioflow", "tigon", "trajectorynet"]
EMBEDDINGS = ["X_pca_scaled", "DM_EigenVectors_scaled"]


def write_manifest(path: Path, methods: list[str]):
    rows = []
    task_id = 0
    for method in methods:
        for emb in EMBEDDINGS:
            outdir = LOGS_DIR / method / emb  # per-seed dirs created inside the SLURM task
            rows.append((task_id, method, emb, str(outdir)))
            task_id += 1
    with open(path, "w") as fh:
        fh.write("task_id\tmethod\tembedding\toutdir\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    print(f"Wrote {path}  ({len(rows)} rows, {len(methods)} methods × {len(EMBEDDINGS)} embeddings)")
    return rows


def main():
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    t1 = write_manifest(SCRIPTS_DIR / "manifest_tier1_grouped.tsv", TIER1_TRAINABLE)
    t2 = write_manifest(SCRIPTS_DIR / "manifest_tier2_grouped.tsv", TIER2_TRAINABLE)
    print(f"\nSLURM array ranges:")
    print(f"  tier 1: 0-{len(t1) - 1}  ({len(t1)} tasks; each runs 3 seeds sequentially)")
    print(f"  tier 2: 0-{len(t2) - 1}  ({len(t2)} tasks; each runs 3 seeds sequentially)")


if __name__ == "__main__":
    main()
