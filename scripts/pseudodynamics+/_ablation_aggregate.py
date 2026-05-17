"""Aggregate the per-seed census into per-arm rows ready for Notion ingest.

Outputs logs/ablation_arm_summary.csv with one row per (dataset, arm).
"""
import csv
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path

ROOT = Path("/rds/user/wz369/hpc-work/pseudodynamics_plus")
CENSUS = ROOT / "logs" / "ablation_census.csv"
OUT = ROOT / "logs" / "ablation_arm_summary.csv"

# Each dataset's nominal max_epochs (from running long-job behaviour).
# Used to flag arms as Partial vs Complete.
NOMINAL_MAX_EPOCHS = {
    "klein_nonOT": 400,
    "klein_OT": 200,
    "tom_pos": 200,
    "synthetic_FP": 400,
}

# A run is "Complete" if best_epoch >= 0.85 * nominal_max_epochs (allows for
# early stopping near the end). "Partial" if it has a checkpoint but
# is below that threshold. "Missing" if no checkpoint at all.
COMPLETE_FRAC = 0.85


def main():
    rows = list(csv.DictReader(CENSUS.open()))
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[(r["dataset"], r["arm"])].append(r)

    out_rows = []
    for (ds, arm), seed_rows in sorted(by_arm.items()):
        nominal_max = NOMINAL_MAX_EPOCHS[ds]
        complete_thresh = int(nominal_max * COMPLETE_FRAC)

        # Pull weights from the V0 config (all 3 seeds share it)
        v0 = seed_rows[0]
        weights = {
            "lambda_D": float(v0["v0_lambda_D"]) if v0["v0_lambda_D"] else None,
            "lambda_v": float(v0["v0_lambda_v"]) if v0["v0_lambda_v"] else None,
            "lambda_g": float(v0["v0_lambda_g"]) if v0["v0_lambda_g"] else None,
            "lambda_CFM": float(v0["v0_lambda_CFM"]) if v0["v0_lambda_CFM"] else None,
            "lambda_NeuralODE": float(v0["v0_lambda_NeuralODE"]) if v0["v0_lambda_NeuralODE"] else None,
            "lambda_R": float(v0["v0_lambda_R"]) if v0["v0_lambda_R"] else None,
        }

        # Per-seed status
        seed_status = {}
        epochs = []
        val_losses = []
        versions = []
        run_dates = []
        ck_dirs = []
        mismatches = []
        for r in seed_rows:
            seed = int(r["seed"])
            if not r["checkpoint_path"]:
                seed_status[seed] = "Missing"
                continue
            ep = int(r["best_epoch"])
            vl = float(r["best_val_loss"])
            epochs.append(ep)
            val_losses.append(vl)
            versions.append(r["version"])
            if r["run_date"]:
                run_dates.append(r["run_date"])
            ck_dirs.append(str(Path(r["checkpoint_path"]).parent))
            if r["weight_mismatch"]:
                mismatches.append(f"seed{seed}: {r['weight_mismatch']}")
            seed_status[seed] = "Complete" if ep >= complete_thresh else "Partial"

        seeds_done = sum(1 for s in seed_status.values() if s == "Complete")
        seeds_partial = sum(1 for s in seed_status.values() if s == "Partial")
        seeds_missing = sum(1 for s in seed_status.values() if s == "Missing")

        # Roll-up arm status
        if seeds_missing == 3:
            arm_status = "Missing"
        elif seeds_done == 3:
            arm_status = "Complete"
        elif seeds_done + seeds_partial == 3 and seeds_done >= 1:
            arm_status = "Mixed"
        elif seeds_done == 0 and seeds_partial > 0:
            arm_status = "Partial"
        else:
            arm_status = "Failed"

        out_rows.append({
            "dataset": ds,
            "arm": arm,
            "lambda_D": weights["lambda_D"],
            "lambda_v": weights["lambda_v"],
            "lambda_g": weights["lambda_g"],
            "lambda_CFM": weights["lambda_CFM"],
            "lambda_NeuralODE": weights["lambda_NeuralODE"],
            "lambda_R": weights["lambda_R"],
            "seeds_complete": seeds_done,
            "seeds_partial": seeds_partial,
            "seeds_missing": seeds_missing,
            "status": arm_status,
            "nominal_max_epochs": nominal_max,
            "median_best_epoch": int(stats.median(epochs)) if epochs else None,
            "min_best_epoch": min(epochs) if epochs else None,
            "max_best_epoch": max(epochs) if epochs else None,
            "median_val_loss": round(stats.median(val_losses), 4) if val_losses else None,
            "min_val_loss": round(min(val_losses), 4) if val_losses else None,
            "max_val_loss": round(max(val_losses), 4) if val_losses else None,
            "versions": ",".join(versions),
            "run_date": min(run_dates).split(" ")[0] if run_dates else None,
            "checkpoint_dir": ck_dirs[0] if ck_dirs else "",  # representative seed_0 dir
            "all_seed_dirs": "; ".join(ck_dirs),
            "weight_mismatch": "; ".join(mismatches),
        })

    fieldnames = list(out_rows[0].keys())
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {len(out_rows)} arm rows to {OUT}")
    # Print compact status grid
    print()
    print(f"{'Dataset':<14}{'Arm':<24}{'Status':<10}{'C/P/M':<8}{'med_ep':<8}{'med_val':<10}")
    for r in out_rows:
        cpm = f"{r['seeds_complete']}/{r['seeds_partial']}/{r['seeds_missing']}"
        print(f"{r['dataset']:<14}{r['arm']:<24}{r['status']:<10}{cpm:<8}"
              f"{str(r['median_best_epoch'] or '-'):<8}{str(r['median_val_loss'] or '-'):<10}")


if __name__ == "__main__":
    main()
