"""Emit JSON payloads for notion-create-pages from logs/ablation_arm_summary.csv.

Two outputs:
  logs/notion_main_pages.json    -> 48 real-data rows
  logs/notion_synth_pages.json   -> 4  synthetic_FP rows
"""
import csv
import json
from pathlib import Path

ROOT = Path("/rds/user/wz369/hpc-work/pseudodynamics_plus")
CSV_PATH = ROOT / "logs" / "ablation_arm_summary.csv"

ARM_FROM_NAME = {
    "baseline": "baseline",
    "lambdaD": "lambdaD",
    "lambdav": "lambdav",
    "lambdag": "lambdag",
    "lambdaCFM": "lambdaCFM",
    "lambdaNeuralODE": "lambdaNeuralODE",
    "lambdaR": "lambdaR",
}


def parse_arm(arm_str):
    """'lambdaD_0.01' -> ('lambdaD', '0.01'); 'baseline' -> ('baseline', 'N/A')"""
    if arm_str == "baseline":
        return "baseline", "N/A"
    parts = arm_str.split("_", 1)
    return parts[0], parts[1] if len(parts) > 1 else "N/A"


def to_num(v):
    if v in (None, "", "None"):
        return None
    try:
        f = float(v)
        return f
    except ValueError:
        return None


def main():
    rows = list(csv.DictReader(CSV_PATH.open()))
    main_pages = []
    synth_pages = []

    for r in rows:
        ds = r["dataset"]
        arm_name, sweep_val = parse_arm(r["arm"])
        cpm = f"{r['seeds_complete']}/{r['seeds_partial']}/{r['seeds_missing']}"

        if ds == "synthetic_FP":
            page = {
                "properties": {
                    "Name": f"synthetic_FP / {r['arm']}",
                    "Arm": r["arm"],
                    "lambda_D": to_num(r["lambda_D"]),
                    "Status": "Pending data",  # all 12 missing
                    "Seeds complete": to_num(r["seeds_complete"]),
                    "Median best epoch": to_num(r["median_best_epoch"]),
                    "Median val_loss": to_num(r["median_val_loss"]),
                    "Eval status": "Pending",
                    "Checkpoint dir": r["checkpoint_dir"] or "logs/synthetic_FP_ablation/<arm>/seed_<S>/pde_params_tsense (not yet populated)",
                    "Blocker": "data/synthetic_FP.h5ad missing — 06_synthetic_FP_simulator.py has not been run; 05_train_ablation_synthetic_FP.sh not launched.",
                    "Notes": "Feeds Supp Fig 2 (R1.4a). Sprint 2 plan flags this as 'roll to Sprint 3 if data not generated'.",
                },
            }
            synth_pages.append(page)
            continue

        # Real-data rows
        notes_parts = []
        if r["status"] == "Missing":
            notes_parts.append("All 3 seeds crashed before checkpoint save (Singularity 1m timeout / context deadline).")
        elif r["status"] in ("Partial", "Mixed"):
            notes_parts.append(
                f"Median best epoch {r['median_best_epoch']} vs nominal max {r['nominal_max_epochs']}; "
                f"some seeds early-stopped or crashed mid-training."
            )
        if ds == "klein_OT" and arm_name == "lambdav":
            notes_parts.append("klein_OT does not sweep lambda_v (OT-CFM replaces velocity term).")
        if ds == "klein_nonOT" and arm_name == "lambdaCFM":
            notes_parts.append("klein_nonOT does not use CFM — arm intentionally absent.")
        if ds == "tom_pos" and arm_name == "lambdaCFM":
            notes_parts.append("tom_pos uses non-OT path — no CFM arm.")

        page = {
            "properties": {
                "Name": f"{ds} / {r['arm']}",
                "Dataset": ds,
                "Arm": arm_name,
                "Sweep value": sweep_val,
                "lambda_D": to_num(r["lambda_D"]),
                "lambda_v": to_num(r["lambda_v"]),
                "lambda_g": to_num(r["lambda_g"]),
                "lambda_CFM": to_num(r["lambda_CFM"]),
                "lambda_NeuralODE": to_num(r["lambda_NeuralODE"]),
                "lambda_R": to_num(r["lambda_R"]),
                "Status": r["status"],
                "Seeds C/P/M": cpm,
                "Median best epoch": to_num(r["median_best_epoch"]),
                "Min best epoch": to_num(r["min_best_epoch"]),
                "Median val_loss": to_num(r["median_val_loss"]),
                "Min val_loss": to_num(r["min_val_loss"]),
                "Eval status": "Pending eval",
                "Slurm versions": r["versions"],
                "Checkpoint dir": r["checkpoint_dir"] or "(no checkpoint saved)",
                "Notes": " ".join(notes_parts),
            },
        }
        # Date
        if r["run_date"]:
            page["properties"]["date:Run date:start"] = r["run_date"]
            page["properties"]["date:Run date:is_datetime"] = 0
        main_pages.append(page)

    main_path = ROOT / "logs" / "notion_main_pages.json"
    synth_path = ROOT / "logs" / "notion_synth_pages.json"
    main_path.write_text(json.dumps(main_pages, indent=2, default=str))
    synth_path.write_text(json.dumps(synth_pages, indent=2, default=str))
    print(f"Main DB pages:     {len(main_pages):>3}  -> {main_path}")
    print(f"Synthetic DB pages:{len(synth_pages):>3}  -> {synth_path}")


if __name__ == "__main__":
    main()
