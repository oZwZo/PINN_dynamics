"""Inventory ablation runs: arm × seed → best checkpoint, version, run config, weights.

Outputs CSV at logs/ablation_census.csv with one row per (dataset, arm, seed).
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path("/rds/user/wz369/hpc-work/pseudodynamics_plus")
MANIFEST = ROOT / "logs" / "ablation_manifest.txt"
OUT_CSV = ROOT / "logs" / "ablation_census.csv"

WEIGHT_KEYS = [
    "D_penalty",        # λ_D
    "deltax_weight",    # λ_v
    "growth_weight",    # λ_g
    "cfm_weight",       # λ_CFM
    "neuralode_weight", # λ_NeuralODE
    "R_weight",         # λ_R
    "weight_intensity",
    "D_var_weight",
]

CKPT_RE = re.compile(r"epoch=(\d+)-val_loss=(-?[0-9.]+)\.ckpt")


def parse_v0(p: Path) -> dict:
    with p.open() as f:
        return json.load(f)["raw_args"]


def parse_run_config(p: Path) -> dict:
    """Parse the per-run V<jobid>_config.json (newer format with experiment_config etc.)."""
    with p.open() as f:
        d = json.load(f)
    flat = {}
    flat["run_date"] = d.get("run_date")
    ec = d.get("experiment_config", {})
    flat["version"] = ec.get("version")
    flat["save_dir"] = ec.get("save_dir")
    flat["checkpoint_dir"] = ec.get("checkpoint_dir")
    mc = d.get("model_config", {})
    for k in WEIGHT_KEYS:
        flat[k] = mc.get(k)
    flat["max_epochs"] = d.get("trainer_config", {}).get("max_epochs")
    return flat


def best_ckpt(seed_dir: Path) -> Tuple[Optional[Path], Optional[int], Optional[float], Optional[int]]:
    """Return (path, epoch, val_loss, version) of the best checkpoint under a seed dir."""
    lightning_root = seed_dir / "pde_params_tsense" / "lightning_logs"
    if not lightning_root.is_dir():
        return None, None, None, None
    candidates = []  # (epoch, val_loss, path, version)
    for vdir in lightning_root.glob("version_*"):
        vid = int(vdir.name.split("_", 1)[1])
        ck_dir = vdir / "checkpoints"
        if not ck_dir.is_dir():
            continue
        for ck in ck_dir.glob("epoch=*-val_loss=*.ckpt"):
            m = CKPT_RE.match(ck.name)
            if not m:
                continue
            ep = int(m.group(1))
            vl = float(m.group(2))
            candidates.append((ep, vl, ck, vid))
    if not candidates:
        return None, None, None, None
    # pick highest-epoch checkpoint; tiebreak on lowest val_loss
    candidates.sort(key=lambda x: (-x[0], x[1]))
    ep, vl, ck, vid = candidates[0]
    return ck, ep, vl, vid


def find_run_config_for_version(seed_dir: Path, version: Optional[int]) -> Optional[Path]:
    if version is None:
        return None
    target = seed_dir / "pde_params_tsense" / f"V{version}_config.json"
    if target.exists():
        return target
    # fallback: pick newest V*_config.json
    cands = sorted((seed_dir / "pde_params_tsense").glob("V*_config.json"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def main():
    rows = []
    with MANIFEST.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            idx, v0_path, dataset, arm = line.split("\t")
            v0 = parse_v0(Path(v0_path))
            arm_dir = Path(v0_path).parent
            for seed in (0, 1, 2):
                seed_dir = arm_dir / f"seed_{seed}"
                ck, ep, vl, vid = best_ckpt(seed_dir)
                run_cfg_path = find_run_config_for_version(seed_dir, vid)
                run = parse_run_config(run_cfg_path) if run_cfg_path else {}
                row = {
                    "manifest_idx": idx,
                    "dataset": dataset,
                    "arm": arm,
                    "seed": seed,
                    "v0_lambda_D": v0.get("D_penalty"),
                    "v0_lambda_v": v0.get("deltax_weight"),
                    "v0_lambda_g": v0.get("growth_weight"),
                    "v0_lambda_CFM": v0.get("cfm_weight"),
                    "v0_lambda_NeuralODE": v0.get("neuralode_weight"),
                    "v0_lambda_R": v0.get("R_weight"),
                    "v0_weight_intensity": v0.get("weight_intensity"),
                    "v0_D_var_weight": v0.get("D_var_weight"),
                    "run_lambda_D": run.get("D_penalty"),
                    "run_lambda_v": run.get("deltax_weight"),
                    "run_lambda_g": run.get("growth_weight"),
                    "run_lambda_CFM": run.get("cfm_weight"),
                    "run_lambda_NeuralODE": run.get("neuralode_weight"),
                    "run_lambda_R": run.get("R_weight"),
                    "run_weight_intensity": run.get("weight_intensity"),
                    "run_D_var_weight": run.get("D_var_weight"),
                    "best_epoch": ep,
                    "best_val_loss": vl,
                    "version": vid,
                    "run_date": run.get("run_date"),
                    "max_epochs": run.get("max_epochs"),
                    "checkpoint_path": str(ck) if ck else "",
                    "run_config_path": str(run_cfg_path) if run_cfg_path else "",
                }
                # Mismatch flags
                mism = []
                for short, k in [("D", "D_penalty"), ("v", "deltax_weight"), ("g", "growth_weight"),
                                 ("CFM", "cfm_weight"), ("NeuralODE", "neuralode_weight"), ("R", "R_weight")]:
                    a = v0.get(k)
                    b = run.get(k)
                    if b is not None and a != b:
                        mism.append(f"λ_{short}:{a}->{b}")
                row["weight_mismatch"] = "; ".join(mism)
                rows.append(row)
    fieldnames = list(rows[0].keys())
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT_CSV}")
    # Summary
    by_status = {"complete": 0, "partial": 0, "missing": 0}
    for r in rows:
        if not r["checkpoint_path"]:
            by_status["missing"] += 1
        elif (r["max_epochs"] is not None and r["best_epoch"] is not None and r["best_epoch"] >= r["max_epochs"] - 5):
            by_status["complete"] += 1
        else:
            by_status["partial"] += 1
    print(f"Status: {by_status}")


if __name__ == "__main__":
    main()
