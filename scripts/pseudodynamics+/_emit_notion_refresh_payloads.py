"""Build a clean refresh payload for the existing Notion registries.

Produces one (page_id, properties) entry per arm with:
  - Status            (re-derived from current census)
  - Seeds C/P/M       (re-derived)
  - Median best epoch (re-derived)
  - Median val_loss   (re-derived)
  - Notes             (auto-generated, supersedes any stale text)
  - Best ckpt seed_0/1/2  + Best epoch seed_0/1/2  (per-seed)
  - Config path
  - Changed param

Outputs:
  logs/notion_refresh_payloads.json
"""
from __future__ import annotations
import csv
import json
import re
import statistics as stats
from collections import defaultdict
from pathlib import Path

ROOT = Path("/rds/user/wz369/hpc-work/pseudodynamics_plus")
CENSUS = ROOT / "logs" / "ablation_census.csv"
MANIFEST = ROOT / "logs" / "ablation_manifest.txt"
OUT = ROOT / "logs" / "notion_refresh_payloads.json"

NOMINAL_MAX_EPOCHS = {"klein_nonOT": 400, "klein_OT": 200, "tom_pos": 200, "synthetic_FP": 400}
COMPLETE_FRAC = 0.85

# Hard-coded {Name -> page_id} from notion-search results
PAGE_IDS = {
    "klein_OT / baseline": "35770f33-0647-81e5-987b-cea3c1a87269",
    "klein_OT / lambdaCFM_0": "35770f33-0647-8129-8e79-f085d0302bb0",
    "klein_OT / lambdaCFM_0.01": "35770f33-0647-81fd-bfdb-f5928539f4ed",
    "klein_OT / lambdaCFM_10": "35770f33-0647-8126-8bff-fe270b4ea608",
    "klein_OT / lambdaD_0": "35770f33-0647-8189-b7de-d8a7aa202da3",
    "klein_OT / lambdaD_0.01": "35770f33-0647-81c7-ab06-ed5b7825ef79",
    "klein_OT / lambdaD_10": "35770f33-0647-8130-933e-e6810210a747",
    "klein_OT / lambdaNeuralODE_0": "35770f33-0647-810f-b95d-cca5f8b03ba7",
    "klein_OT / lambdaNeuralODE_0.01": "35770f33-0647-8177-968a-cb3b919404c1",
    "klein_OT / lambdaNeuralODE_10": "35770f33-0647-8132-873c-dc6e18219f0f",
    "klein_OT / lambdaR_0": "35770f33-0647-8183-8eb6-d5958317798b",
    "klein_OT / lambdaR_100": "35770f33-0647-8164-a853-fa4e16e732df",
    "klein_OT / lambdaR_1000": "35770f33-0647-8166-8c87-ebb73f0e06f6",
    "klein_OT / lambdag_0": "35770f33-0647-8108-a076-f7b2e21d751c",
    "klein_OT / lambdag_0.01": "35770f33-0647-8105-846f-dc37495c1d87",
    "klein_OT / lambdag_10": "35770f33-0647-8157-8977-d2ebd8d2a530",
    "klein_nonOT / baseline": "35770f33-0647-8178-a235-d8c1f643574a",
    "klein_nonOT / lambdaD_0": "35770f33-0647-81ad-a214-c52954901402",
    "klein_nonOT / lambdaD_0.01": "35770f33-0647-81ab-90be-e25bcb2d3926",
    "klein_nonOT / lambdaD_10": "35770f33-0647-8138-b77a-ea368053b6b4",
    "klein_nonOT / lambdaNeuralODE_0": "35770f33-0647-81f4-80f2-c5f77b6fe76c",
    "klein_nonOT / lambdaNeuralODE_0.01": "35770f33-0647-81ce-8121-fbab15eef023",
    "klein_nonOT / lambdaNeuralODE_10": "35770f33-0647-8112-b185-e032b276aaf2",
    "klein_nonOT / lambdaR_0": "35770f33-0647-81f5-b61a-c6d2d18b163b",
    "klein_nonOT / lambdaR_100": "35770f33-0647-8116-92fe-c9486d2964ba",
    "klein_nonOT / lambdaR_1000": "35770f33-0647-81d6-aacf-e3bd7ac966e7",
    "klein_nonOT / lambdag_0": "35770f33-0647-81fc-8b80-edca2f2d9f1c",
    "klein_nonOT / lambdag_0.01": "35770f33-0647-817d-9f42-c2a89df46772",
    "klein_nonOT / lambdag_10": "35770f33-0647-8126-bf29-c41cd22c2fd6",
    "klein_nonOT / lambdav_0": "35770f33-0647-816d-9c18-df9e29612fdb",
    "klein_nonOT / lambdav_0.01": "35770f33-0647-81bf-9128-ca92da0fe6cd",
    "klein_nonOT / lambdav_10": "35770f33-0647-81b3-a6f8-d3579e9cb8d8",
    "tom_pos / baseline": "35770f33-0647-8121-89b5-cdcf06d7cf49",
    "tom_pos / lambdaD_0": "35770f33-0647-81eb-832b-ce18434e46aa",
    "tom_pos / lambdaD_0.01": "35770f33-0647-81d4-a35b-e5eaa71a5d39",
    "tom_pos / lambdaD_10": "35770f33-0647-8115-8899-ec0043d7d2e5",
    "tom_pos / lambdaNeuralODE_0": "35770f33-0647-8128-a02b-f9660d869d4b",
    "tom_pos / lambdaNeuralODE_0.01": "35770f33-0647-8110-a1bb-dfaeca88220f",
    "tom_pos / lambdaNeuralODE_10": "35770f33-0647-8185-be88-c98cbbaa5ddd",
    "tom_pos / lambdaR_0": "35770f33-0647-8141-8402-c7d1a02924e3",
    "tom_pos / lambdaR_100": "35770f33-0647-8164-938e-de5ad2b210dd",
    "tom_pos / lambdaR_1000": "35770f33-0647-8197-a157-da02254985e6",
    "tom_pos / lambdag_0": "35770f33-0647-8103-aa0b-d1e82d82f135",
    "tom_pos / lambdag_0.01": "35770f33-0647-812f-ae3e-c50f125fdcc7",
    "tom_pos / lambdag_10": "35770f33-0647-81ec-a1b7-f05b7e77f2bf",
    "tom_pos / lambdav_0": "35770f33-0647-81dd-b7a3-da9f5df57b73",
    "tom_pos / lambdav_0.01": "35770f33-0647-8103-b3b6-dcd6790dd7c0",
    "tom_pos / lambdav_10": "35770f33-0647-8156-8a1c-e5ca1f6e01f1",
    "synthetic_FP / baseline": "35770f33-0647-8125-b987-e1a28b89e291",
    "synthetic_FP / lambdaD_0": "35770f33-0647-81d2-8896-e487b7e55bc5",
    "synthetic_FP / lambdaD_0.01": "35770f33-0647-8140-a531-c475d23fcd74",
    "synthetic_FP / lambdaD_10": "35770f33-0647-8128-8de4-f84cb59d9cca",
}

# Arms whose SLURM resubmits (28886824, 28886826) are still running on May 7
IN_FLIGHT_REAL = {
    "klein_nonOT / lambdav_0": "28886824_4 (36h budget)",
    "klein_OT / lambdaCFM_0": "28886826_4 (36h budget)",
    "klein_OT / lambdaNeuralODE_0": "28886826_10 (36h budget)",
    "klein_OT / lambdaR_1000": "28886826_15 (36h budget)",
}
IN_FLIGHT_SYNTHETIC = {"synthetic_FP / baseline"}

CKPT_RE_NEG = re.compile(r"epoch=(\d+)-val_loss=(-?[0-9.]+)\.ckpt")


def parse_arm(arm):
    if arm == "baseline":
        return "baseline", None
    n, v = arm.split("_", 1)
    return n, v


def changed_param(arm):
    n, v = parse_arm(arm)
    if n == "baseline":
        return "baseline (default lambdas: D=v=g=NeuralODE=R=1, CFM=0; klein_OT swaps v->CFM)"
    short = {"lambdaD": "lambda_D", "lambdav": "lambda_v", "lambdag": "lambda_g",
             "lambdaCFM": "lambda_CFM", "lambdaNeuralODE": "lambda_NeuralODE", "lambdaR": "lambda_R"}[n]
    return f"{short} = {v}"


def scan_synthetic_seed(seed_dir: Path):
    lroot = seed_dir / "pde_params_tsense" / "lightning_logs"
    if not lroot.is_dir():
        return None, None
    best = None
    for vdir in lroot.glob("version_*"):
        ck_dir = vdir / "checkpoints"
        if not ck_dir.is_dir():
            continue
        for ck in ck_dir.glob("epoch=*-val_loss=*.ckpt"):
            m = CKPT_RE_NEG.match(ck.name)
            if not m:
                continue
            ep = int(m.group(1))
            if best is None or ep > best[0]:
                best = (ep, ck)
    return (best[1] if best else None), (best[0] if best else None)


def derive_status(epochs, partials, missings, nominal):
    """Mirror _ablation_aggregate.py rules."""
    thresh = int(nominal * COMPLETE_FRAC)
    completes = sum(1 for ep in epochs if ep is not None and ep >= thresh)
    partials_n = sum(1 for ep in epochs if ep is not None and ep < thresh)
    missing_n = sum(1 for ep in epochs if ep is None)
    if missing_n == 3:
        return "Missing", completes, partials_n, missing_n
    if completes == 3:
        return "Complete", completes, partials_n, missing_n
    if completes >= 1 and (completes + partials_n) == 3:
        return "Mixed", completes, partials_n, missing_n
    if completes == 0 and partials_n > 0:
        return "Partial", completes, partials_n, missing_n
    return "Failed", completes, partials_n, missing_n


def make_notes(ds, arm, status, epochs, val_med, nominal, in_flight_tag):
    """Generate fresh notes from current census state."""
    thresh = int(nominal * COMPLETE_FRAC)
    eps_str = "/".join(str(e) if e is not None else "—" for e in epochs)
    parts = []

    if in_flight_tag:
        parts.append(f"⚙️ IN FLIGHT — SLURM {in_flight_tag}; ckpt/epoch will advance.")

    # Coverage description
    if status == "Complete":
        parts.append(f"All 3 seeds reached ≥ {thresh}/{nominal} epochs (epochs: {eps_str}).")
    elif status == "Mixed":
        parts.append(f"At least one seed converged, others Partial (epochs: {eps_str}, threshold {thresh}/{nominal}).")
    elif status == "Partial":
        # Distinguish wallclock-bound from non-converged
        if val_med is not None and val_med > 1000:
            parts.append(
                f"Non-converged: val_loss median ≈ {val_med:.0f} (orders of magnitude above sibling arms). "
                f"Best epochs {eps_str}/{nominal}."
            )
        else:
            parts.append(
                f"Wallclock-bound: best epochs {eps_str}/{nominal}, val_loss median {val_med:.2f} "
                f"(within converged range — usable for trend, not for headline numbers)."
            )
    elif status == "Missing":
        parts.append(f"No checkpoints written — needs (re)submit.")

    # Special-case dataset/arm constraints
    if ds == "klein_OT" and arm.startswith("lambdav"):
        parts.append("(klein_OT does not sweep λ_v — OT-CFM replaces velocity term.)")
    if ds in ("klein_nonOT", "tom_pos") and arm.startswith("lambdaCFM"):
        parts.append("(non-OT path — no CFM arm.)")
    if ds == "synthetic_FP" and arm != "baseline":
        parts.append("(synthetic FP only sweeps λ_D; runs serially after baseline finishes on local GPU.)")
    return " ".join(parts)


def main():
    rows = list(csv.DictReader(CENSUS.open()))
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[(r["dataset"], r["arm"])].append(r)

    # V0 config paths
    v0 = {}
    for line in MANIFEST.open():
        line = line.rstrip("\n")
        if not line:
            continue
        idx, p, ds, arm = line.split("\t")
        v0[(ds, arm)] = p

    payloads = []
    for (ds, arm), seed_rows in sorted(by_arm.items()):
        seed_rows.sort(key=lambda r: int(r["seed"]))
        nominal = NOMINAL_MAX_EPOCHS[ds]

        ckpts = {0: None, 1: None, 2: None}
        epochs = {0: None, 1: None, 2: None}
        vlosses = []
        for r in seed_rows:
            seed = int(r["seed"])
            ck = r["checkpoint_path"]
            if ds == "synthetic_FP":
                seed_dir = Path(v0[(ds, arm)]).parent / f"seed_{seed}"
                ck_real, ep_real = scan_synthetic_seed(seed_dir)
                ckpts[seed] = str(ck_real) if ck_real else None
                epochs[seed] = ep_real
                # extract val_loss from filename
                if ck_real:
                    m = CKPT_RE_NEG.match(ck_real.name)
                    if m:
                        vlosses.append(float(m.group(2)))
            else:
                ckpts[seed] = ck if ck else None
                epochs[seed] = int(r["best_epoch"]) if r["best_epoch"] else None
                if r["best_val_loss"]:
                    vlosses.append(float(r["best_val_loss"]))

        ep_list = [epochs[i] for i in (0, 1, 2)]
        status, c, p, m = derive_status(ep_list, None, None, nominal)
        med_ep = stats.median([e for e in ep_list if e is not None]) if any(e is not None for e in ep_list) else None
        med_vl = stats.median(vlosses) if vlosses else None

        name = f"{ds} / {arm}"
        page_id = PAGE_IDS.get(name)
        if not page_id:
            print(f"  WARN: no page_id for {name}")
            continue

        in_flight_tag = IN_FLIGHT_REAL.get(name) or ("local gpu-q-33 (synthetic_FP serial loop)" if name in IN_FLIGHT_SYNTHETIC else None)
        notes = make_notes(ds, arm, status, ep_list, med_vl, nominal, in_flight_tag)

        # Map synthetic_FP status → that DB's status options
        status_for_synth = {"Complete": "Complete", "Mixed": "Partial", "Partial": "Partial", "Missing": "Pending training", "Failed": "Failed"}
        final_status = status_for_synth.get(status, status) if ds == "synthetic_FP" else status

        props = {
            "Status": final_status,
            "Median best epoch": int(med_ep) if med_ep is not None else None,
            "Median val_loss": round(med_vl, 4) if med_vl is not None else None,
            "Notes": notes,
            "Config path": v0[(ds, arm)],
            "Changed param": changed_param(arm),
            "Best ckpt seed_0": ckpts[0] or "(no-ckpt)",
            "Best ckpt seed_1": ckpts[1] or "(no-ckpt)",
            "Best ckpt seed_2": ckpts[2] or "(no-ckpt)",
            "Best epoch seed_0": epochs[0],
            "Best epoch seed_1": epochs[1],
            "Best epoch seed_2": epochs[2],
        }
        if ds != "synthetic_FP":
            props["Seeds C/P/M"] = f"{c}/{p}/{m}"
        else:
            props["Seeds complete"] = c

        payloads.append({"page_id": page_id, "name": name, "dataset": ds, "properties": props})

    OUT.write_text(json.dumps(payloads, indent=2, default=str))
    print(f"Wrote {len(payloads)} refresh payloads to {OUT}")


if __name__ == "__main__":
    main()
