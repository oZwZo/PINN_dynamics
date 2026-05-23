#!/usr/bin/env python
"""01c_fix_pop_uns.py — In-place fix: move `variance_source` out of `uns['pop']`.

The pseudodynamics+ reader iterates `uns['pop']` and indexes every value by the
timepoint slice; a string value crashes it with TypeError. This patcher relocates
the annotation to `uns['pop_meta']` so the reader is happy.

Idempotent.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import scanpy as sc

ROOT = Path("/rds/user/wz369/hpc-work/PINN_dynamics")
DEFAULT = ROOT / "data" / "cordblood_addpop.h5ad"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_path", default=str(DEFAULT))
    args = p.parse_args()

    path = Path(args.data_path)
    print(f"Loading {path} ...")
    adata = sc.read_h5ad(path)

    pop = dict(adata.uns.get("pop", {}))
    moved = False
    if "variance_source" in pop:
        adata.uns.setdefault("pop_meta", {})
        adata.uns["pop_meta"]["variance_source"] = pop.pop("variance_source")
        moved = True
    # Strip any other non-indexable fields just in case
    for k in list(pop.keys()):
        v = pop[k]
        try:
            v[0]
            float(len(v))
        except (TypeError, IndexError):
            print(f"  Moving non-indexable uns['pop']['{k}'] → uns['pop_meta']")
            adata.uns.setdefault("pop_meta", {})[k] = pop.pop(k)
            moved = True

    if not moved:
        print("  uns['pop'] already clean — no changes.")
        return
    adata.uns["pop"] = pop
    print(f"  Final uns['pop'] keys: {sorted(pop.keys())}")
    print(f"  uns['pop_meta'] keys: {sorted(adata.uns.get('pop_meta', {}).keys())}")
    print(f"Writing {path} ...")
    adata.write_h5ad(path)
    print("Done.")


if __name__ == "__main__":
    main()
