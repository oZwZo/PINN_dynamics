"""Build per-arm enrichment payload for the existing Notion registries.

Per arm we emit:
  - Config path (V0_config.json absolute path)
  - Changed param (human-readable, e.g. 'lambda_D = 0.01' or 'baseline (defaults)')
  - Best ckpt seed_0 / seed_1 / seed_2 (absolute path or 'no-ckpt' or 'in-progress')
  - Best epoch seed_0 / seed_1 / seed_2 (int or None)

Outputs:
  logs/notion_enrichment.json     (52 entries, keyed by 'dataset/arm')

Synthetic_FP arms re-scan the lightning_logs dirs directly because the
shared census regex doesn't accept negative val_loss values.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path("/rds/user/wz369/hpc-work/pseudodynamics_plus")
CENSUS = ROOT / "logs" / "ablation_census.csv"
MANIFEST = ROOT / "logs" / "ablation_manifest.txt"
OUT_JSON = ROOT / "logs" / "notion_enrichment.json"

# epoch=<int>-val_loss=<float, optional minus>.ckpt
CKPT_RE = re.compile(r"epoch=(\d+)-val_loss=(-?[0-9.]+)\.ckpt")


def parse_arm(arm_str):
    """'lambdaD_0.01' -> ('lambdaD', '0.01'); 'baseline' -> ('baseline', None)."""
    if arm_str == "baseline":
        return "baseline", None
    parts = arm_str.split("_", 1)
    return parts[0], parts[1] if len(parts) > 1 else None


def changed_param_str(arm_str):
    name, val = parse_arm(arm_str)
    if name == "baseline":
        return "baseline (default lambdas: D=v=g=NeuralODE=R=1, CFM=0; klein_OT swaps v->CFM)"
    short = {
        "lambdaD": "lambda_D",
        "lambdav": "lambda_v",
        "lambdag": "lambda_g",
        "lambdaCFM": "lambda_CFM",
        "lambdaNeuralODE": "lambda_NeuralODE",
        "lambdaR": "lambda_R",
    }[name]
    return f"{short} = {val}"


def scan_synthetic_seed(seed_dir: Path):
    """Return (best_ckpt_path or None, best_epoch or None) for synthetic_FP, where val_loss may be negative."""
    lroot = seed_dir / "pde_params_tsense" / "lightning_logs"
    if not lroot.is_dir():
        return None, None
    candidates = []
    for vdir in lroot.glob("version_*"):
        ck_dir = vdir / "checkpoints"
        if not ck_dir.is_dir():
            continue
        for ck in ck_dir.glob("epoch=*-val_loss=*.ckpt"):
            m = CKPT_RE.match(ck.name)
            if not m:
                continue
            ep = int(m.group(1))
            candidates.append((ep, ck))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: -x[0])
    ep, ck = candidates[0]
    return str(ck), ep


def main():
    rows = list(csv.DictReader(CENSUS.open()))
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[(r["dataset"], r["arm"])].append(r)

    # also need V0_config paths from manifest
    v0_paths = {}
    for line in MANIFEST.open():
        line = line.rstrip("\n")
        if not line:
            continue
        idx, v0_path, dataset, arm = line.split("\t")
        v0_paths[(dataset, arm)] = v0_path

    enrichment = {}
    for (ds, arm), seed_rows in sorted(by_arm.items()):
        seed_rows.sort(key=lambda r: int(r["seed"]))
        cfg_path = v0_paths.get((ds, arm), "")
        # Per-seed ckpt + epoch (filling missing with explicit string)
        ckpts = {0: None, 1: None, 2: None}
        epochs = {0: None, 1: None, 2: None}
        for r in seed_rows:
            seed = int(r["seed"])
            ck = r["checkpoint_path"]
            ep_str = r["best_epoch"]
            if ds == "synthetic_FP":
                # re-scan because census regex misses negative val_loss
                seed_dir = Path(cfg_path).parent / f"seed_{seed}"
                ck_real, ep_real = scan_synthetic_seed(seed_dir)
                ckpts[seed] = ck_real or "(no-ckpt)"
                epochs[seed] = ep_real
            else:
                ckpts[seed] = ck or "(no-ckpt)"
                epochs[seed] = int(ep_str) if ep_str else None
        key = f"{ds}/{arm}"
        enrichment[key] = {
            "name": f"{ds} / {arm}",
            "config_path": cfg_path,
            "changed_param": changed_param_str(arm),
            "ckpt_seed_0": ckpts[0],
            "ckpt_seed_1": ckpts[1],
            "ckpt_seed_2": ckpts[2],
            "epoch_seed_0": epochs[0],
            "epoch_seed_1": epochs[1],
            "epoch_seed_2": epochs[2],
        }

    OUT_JSON.write_text(json.dumps(enrichment, indent=2, default=str))
    print(f"Wrote {len(enrichment)} entries to {OUT_JSON}")
    # Quick summary
    for k, v in enrichment.items():
        eps = [v["epoch_seed_0"], v["epoch_seed_1"], v["epoch_seed_2"]]
        print(f"  {k:<40} param={v['changed_param']:<45} epochs={eps}")


if __name__ == "__main__":
    main()
