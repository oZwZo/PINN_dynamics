"""Emit per-page Notion update payloads for the registry enrichment.

Combines logs/notion_enrichment.json with hard-coded {Name -> page_id} maps
produced from notion-search, and prints a JSON list of {page_id, properties}
ready to feed to mcp notion-update-page.
"""
import json
from pathlib import Path

ROOT = Path("/rds/user/wz369/hpc-work/pseudodynamics_plus")

# Name -> page_id, gathered from notion-search results on May 7
PAGE_IDS = {
    # klein_OT
    "klein_OT / baseline":               "35770f33-0647-81e5-987b-cea3c1a87269",
    "klein_OT / lambdaCFM_0":            "35770f33-0647-8129-8e79-f085d0302bb0",
    "klein_OT / lambdaCFM_0.01":         "35770f33-0647-81fd-bfdb-f5928539f4ed",
    "klein_OT / lambdaCFM_10":           "35770f33-0647-8126-8bff-fe270b4ea608",
    "klein_OT / lambdaD_0":              "35770f33-0647-8189-b7de-d8a7aa202da3",
    "klein_OT / lambdaD_0.01":           "35770f33-0647-81c7-ab06-ed5b7825ef79",
    "klein_OT / lambdaD_10":             "35770f33-0647-8130-933e-e6810210a747",
    "klein_OT / lambdaNeuralODE_0":      "35770f33-0647-810f-b95d-cca5f8b03ba7",
    "klein_OT / lambdaNeuralODE_0.01":   "35770f33-0647-8177-968a-cb3b919404c1",
    "klein_OT / lambdaNeuralODE_10":     "35770f33-0647-8132-873c-dc6e18219f0f",
    "klein_OT / lambdaR_0":              "35770f33-0647-8183-8eb6-d5958317798b",
    "klein_OT / lambdaR_100":            "35770f33-0647-8164-a853-fa4e16e732df",
    "klein_OT / lambdaR_1000":           "35770f33-0647-8166-8c87-ebb73f0e06f6",
    "klein_OT / lambdag_0":              "35770f33-0647-8108-a076-f7b2e21d751c",
    "klein_OT / lambdag_0.01":           "35770f33-0647-8105-846f-dc37495c1d87",
    "klein_OT / lambdag_10":             "35770f33-0647-8157-8977-d2ebd8d2a530",
    # klein_nonOT
    "klein_nonOT / baseline":            "35770f33-0647-8178-a235-d8c1f643574a",
    "klein_nonOT / lambdaD_0":           "35770f33-0647-81ad-a214-c52954901402",
    "klein_nonOT / lambdaD_0.01":        "35770f33-0647-81ab-90be-e25bcb2d3926",
    "klein_nonOT / lambdaD_10":          "35770f33-0647-8138-b77a-ea368053b6b4",
    "klein_nonOT / lambdaNeuralODE_0":   "35770f33-0647-81f4-80f2-c5f77b6fe76c",
    "klein_nonOT / lambdaNeuralODE_0.01": "35770f33-0647-81ce-8121-fbab15eef023",
    "klein_nonOT / lambdaNeuralODE_10":  "35770f33-0647-8112-b185-e032b276aaf2",
    "klein_nonOT / lambdaR_0":           "35770f33-0647-81f5-b61a-c6d2d18b163b",
    "klein_nonOT / lambdaR_100":         "35770f33-0647-8116-92fe-c9486d2964ba",
    "klein_nonOT / lambdaR_1000":        "35770f33-0647-81d6-aacf-e3bd7ac966e7",
    "klein_nonOT / lambdag_0":           "35770f33-0647-81fc-8b80-edca2f2d9f1c",
    "klein_nonOT / lambdag_0.01":        "35770f33-0647-817d-9f42-c2a89df46772",
    "klein_nonOT / lambdag_10":          "35770f33-0647-8126-bf29-c41cd22c2fd6",
    "klein_nonOT / lambdav_0":           "35770f33-0647-816d-9c18-df9e29612fdb",
    "klein_nonOT / lambdav_0.01":        "35770f33-0647-81bf-9128-ca92da0fe6cd",
    "klein_nonOT / lambdav_10":          "35770f33-0647-81b3-a6f8-d3579e9cb8d8",
    # tom_pos
    "tom_pos / baseline":                "35770f33-0647-8121-89b5-cdcf06d7cf49",
    "tom_pos / lambdaD_0":               "35770f33-0647-81eb-832b-ce18434e46aa",
    "tom_pos / lambdaD_0.01":            "35770f33-0647-81d4-a35b-e5eaa71a5d39",
    "tom_pos / lambdaD_10":              "35770f33-0647-8115-8899-ec0043d7d2e5",
    "tom_pos / lambdaNeuralODE_0":       "35770f33-0647-8128-a02b-f9660d869d4b",
    "tom_pos / lambdaNeuralODE_0.01":    "35770f33-0647-8110-a1bb-dfaeca88220f",
    "tom_pos / lambdaNeuralODE_10":      "35770f33-0647-8185-be88-c98cbbaa5ddd",
    "tom_pos / lambdaR_0":               "35770f33-0647-8141-8402-c7d1a02924e3",
    "tom_pos / lambdaR_100":             "35770f33-0647-8164-938e-de5ad2b210dd",
    "tom_pos / lambdaR_1000":            "35770f33-0647-8197-a157-da02254985e6",
    "tom_pos / lambdag_0":               "35770f33-0647-8103-aa0b-d1e82d82f135",
    "tom_pos / lambdag_0.01":            "35770f33-0647-812f-ae3e-c50f125fdcc7",
    "tom_pos / lambdag_10":              "35770f33-0647-81ec-a1b7-f05b7e77f2bf",
    "tom_pos / lambdav_0":               "35770f33-0647-81dd-b7a3-da9f5df57b73",
    "tom_pos / lambdav_0.01":            "35770f33-0647-8103-b3b6-dcd6790dd7c0",
    "tom_pos / lambdav_10":              "35770f33-0647-8156-8a1c-e5ca1f6e01f1",
    # synthetic_FP
    "synthetic_FP / baseline":           "35770f33-0647-8125-b987-e1a28b89e291",
    "synthetic_FP / lambdaD_0":          "35770f33-0647-81d2-8896-e487b7e55bc5",
    "synthetic_FP / lambdaD_0.01":       "35770f33-0647-8140-a531-c475d23fcd74",
    "synthetic_FP / lambdaD_10":         "35770f33-0647-8128-8de4-f84cb59d9cca",
}

# Arms whose 28886... resubmits are still running (May 7)
IN_FLIGHT_ARMS = {
    "klein_nonOT / lambdav_0",
    "klein_OT / lambdaCFM_0",
    "klein_OT / lambdaNeuralODE_0",
    "klein_OT / lambdaR_1000",
    "synthetic_FP / baseline",  # local GPU run
}


def main():
    enrichment = json.loads((ROOT / "logs" / "notion_enrichment.json").read_text())
    payloads = []
    for arm_key, e in enrichment.items():
        name = e["name"]
        page_id = PAGE_IDS.get(name)
        if not page_id:
            print(f"  WARN: no page_id for {name}")
            continue
        props = {
            "Config path": e["config_path"],
            "Changed param": e["changed_param"],
            "Best ckpt seed_0": e["ckpt_seed_0"] or "",
            "Best ckpt seed_1": e["ckpt_seed_1"] or "",
            "Best ckpt seed_2": e["ckpt_seed_2"] or "",
            "Best epoch seed_0": e["epoch_seed_0"],
            "Best epoch seed_1": e["epoch_seed_1"],
            "Best epoch seed_2": e["epoch_seed_2"],
        }
        if name in IN_FLIGHT_ARMS:
            props["Notes"] = f"[In progress on May 7 — ckpt/epoch will advance] " + (
                "synthetic_FP/baseline running on local gpu-q-33; "
                if name == "synthetic_FP / baseline"
                else "SLURM 28886824 / 28886826 still running with 36h budget; "
            )
        payloads.append({"page_id": page_id, "properties": props, "name": name})

    out = ROOT / "logs" / "notion_update_payloads.json"
    out.write_text(json.dumps(payloads, indent=2, default=str))
    print(f"Wrote {len(payloads)} update payloads to {out}")


if __name__ == "__main__":
    main()
