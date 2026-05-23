#!/usr/bin/env python
"""_aggregate.py — Aggregate cord blood benchmark eval CSVs into a leaderboard.

Reads every per-run `eval_combined.csv` under `logs/cordblood/<method>/...`,
enforces a strict long-format schema, prints a coverage matrix, and writes:
    results/cordblood/leaderboard.csv         (mean ± std per (method, embedding, metric))
    results/cordblood/aggregation_errors.log  (per-run schema violations)

Strict schema (each per-run CSV must have these columns, no extras):
    method, embedding, seed, timepoint, metric, value

`metric` values expected (per the plan):
    - fate_accuracy
    - W2_scaled_tp{T}   (one per timepoint T)
    - W2_raw_tp{T}      (one per timepoint T)
    - w2_per_clone_mean (optional summary)

Runs with extras or missing columns are excluded and logged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/rds/user/wz369/hpc-work/PINN_dynamics")
DEFAULT_GLOB_ROOT = ROOT / "logs" / "cordblood"
DEFAULT_RESULTS_DIR = ROOT / "results" / "cordblood"

REQUIRED_COLS = {"method", "embedding", "seed", "timepoint", "metric", "value"}


def collect_run_csvs(glob_root: Path):
    """Find every eval_combined.csv under logs/cordblood/."""
    return sorted(glob_root.rglob("eval_combined.csv"))


def validate_and_load(csv_path: Path):
    """Load one per-run CSV. Returns (df, error_str_or_None)."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return None, f"read error: {e}"
    cols = set(df.columns)
    missing = REQUIRED_COLS - cols
    extras = cols - REQUIRED_COLS
    if missing:
        return None, f"missing columns: {sorted(missing)}"
    if extras:
        # Trim down to required cols — extras are tolerated as long as required exist.
        # The plan says strict on missing but we don't punish extras (e.g. method
        # might add provenance like checkpoint_path). Log a soft warning instead.
        soft_warn = f"WARN: ignored extra cols: {sorted(extras)}"
        df = df[list(REQUIRED_COLS)]
    else:
        soft_warn = None
    # cast types so groupby works
    df["seed"] = df["seed"].astype(int)
    df["timepoint"] = df["timepoint"].astype(float)
    df["value"] = df["value"].astype(float)
    df["method"] = df["method"].astype(str)
    df["embedding"] = df["embedding"].astype(str)
    df["metric"] = df["metric"].astype(str)
    return df, soft_warn


def coverage_matrix(merged: pd.DataFrame):
    """Build a [method × embedding × seed] presence matrix per metric."""
    rows = []
    for (method, embedding), g in merged.groupby(["method", "embedding"]):
        seeds_seen = sorted(g["seed"].unique().tolist())
        n_metrics = g["metric"].nunique()
        rows.append({
            "method": method,
            "embedding": embedding,
            "n_seeds": len(seeds_seen),
            "seeds": ",".join(str(s) for s in seeds_seen),
            "n_unique_metrics": n_metrics,
        })
    return pd.DataFrame(rows)


def leaderboard(merged: pd.DataFrame):
    """mean ± std across seeds per (method, embedding, metric, timepoint)."""
    grp = merged.groupby(["method", "embedding", "metric", "timepoint"])["value"]
    agg = grp.agg(["mean", "std", "count"]).reset_index()
    agg = agg.rename(columns={"mean": "value_mean", "std": "value_std", "count": "n_seeds"})
    # NaN std when count==1 → 0.0 is more informative
    agg["value_std"] = agg["value_std"].fillna(0.0)
    return agg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", default=str(DEFAULT_GLOB_ROOT),
                        help=f"Root of per-run logs (default: {DEFAULT_GLOB_ROOT})")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS_DIR),
                        help=f"Output dir (default: {DEFAULT_RESULTS_DIR})")
    parser.add_argument("--manifest", default=None,
                        help="Optional path to manifest TSV (manifest_tier1.tsv or tier2). "
                             "If provided, the coverage matrix highlights missing cells.")
    args = parser.parse_args()

    glob_root = Path(args.logs)
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    err_log = results_dir / "aggregation_errors.log"
    leaderboard_path = results_dir / "leaderboard.csv"
    coverage_path = results_dir / "coverage_matrix.csv"

    csvs = collect_run_csvs(glob_root)
    print(f"Found {len(csvs)} eval_combined.csv files under {glob_root}")

    errors = []
    soft_warns = []
    dfs = []
    for csv in csvs:
        df, err_or_warn = validate_and_load(csv)
        if df is None:
            errors.append(f"{csv}: {err_or_warn}")
            continue
        if err_or_warn:
            soft_warns.append(f"{csv}: {err_or_warn}")
        # Cross-check method/embedding/seed are constant within the file
        for col in ("method", "embedding", "seed"):
            if df[col].nunique() != 1:
                errors.append(f"{csv}: column '{col}' has multiple values "
                              f"({df[col].unique().tolist()[:5]}); skipping.")
                df = None
                break
        if df is not None:
            dfs.append(df)

    # Write errors log unconditionally (empty if no issues)
    with open(err_log, "w") as fh:
        if errors:
            fh.write("# Errors (runs excluded from leaderboard)\n")
            for e in errors:
                fh.write(e + "\n")
        if soft_warns:
            fh.write("\n# Soft warnings (runs INCLUDED with extras dropped)\n")
            for w in soft_warns:
                fh.write(w + "\n")
        if not (errors or soft_warns):
            fh.write("# No schema issues\n")
    print(f"Wrote {err_log} ({len(errors)} errors, {len(soft_warns)} soft warnings)")

    if not dfs:
        print("No valid per-run CSVs found — leaderboard not generated.")
        return 1

    merged = pd.concat(dfs, ignore_index=True)
    cov = coverage_matrix(merged)
    cov.to_csv(coverage_path, index=False)
    print(f"\nCoverage matrix → {coverage_path}")
    print(cov.to_string(index=False))

    # Manifest cross-check
    if args.manifest and Path(args.manifest).exists():
        manifest = pd.read_csv(args.manifest, sep="\t")
        expected = set(
            (m, e, s) for m, e, s in zip(manifest["method"], manifest["embedding"], manifest["seed"].astype(int))
        )
        actual = set(
            (m, e, s) for m, e, s in zip(merged["method"], merged["embedding"], merged["seed"])
        )
        missing = expected - actual
        if missing:
            print(f"\n** Missing runs (per manifest) **: {len(missing)}")
            for m, e, s in sorted(missing):
                print(f"  - method={m}  embedding={e}  seed={s}")

    lb = leaderboard(merged)
    lb.to_csv(leaderboard_path, index=False)
    print(f"\nLeaderboard → {leaderboard_path}  ({len(lb)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
