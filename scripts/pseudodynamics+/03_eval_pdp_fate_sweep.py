"""
Sequential fate-accuracy evaluation over multiple checkpoints.

Mirrors `03_eval_klein_ablation_sweep.sh` (the W2 sweep) but does the loop
INSIDE a single Python process so the expensive setup is paid exactly once:
  * load adata + F_obs  (once)
  * build x_ref / y_ref (once)
  * fit KNN classifier  (once)
  * for each (dataset, arm, seed):
      - load_from_checkpoint
      - build sim_fn
      - run_fate_evaluation(..., knn=knn_pre)   ← reuses the pre-fit KNN
      - write the same 3 CSVs that 03_eval_pdp_fate.py writes

Reuses fate_eval_pipeline.run_fate_evaluation() unchanged — relies on the
new (backward-compatible) `knn=None` kwarg so the single-model script
continues to work exactly as before.

Idempotent: a (dataset, arm, seed) is skipped if its output CSV exists.

Usage (defaults shown; everything is overridable):
  python scripts/pseudodynamics+/03_eval_pdp_fate_sweep.py \
      --datasets klein_OT klein_nonOT \
      --sim_fn ode --t_end_norm 0.2 --noise_scale 1.0 \
      --device cuda:0
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.neighbors import KNeighborsClassifier

os.chdir("/rds/user/wz369/hpc-work/pseudodynamics_plus/")
sys.path.append(os.path.join("/rds/user/wz369/hpc-work/PINN_dynamics", "scripts"))

import pseudodynamics as pdp                       # noqa: E402
import fate_eval_pipeline as fate_eval             # noqa: E402

PDP_DIR = Path("/rds/user/wz369/hpc-work/pseudodynamics_plus")
RES_ROOT = PDP_DIR / "results" / "pseudodynamics+"
CKPT_RE = re.compile(r"epoch=(\d+)-val_loss=(-?[0-9.]+)\.ckpt")


def best_ckpt(version_dir: Path):
    cd = version_dir / "checkpoints"
    if not cd.is_dir():
        return None, -1
    best, best_ep = None, -1
    for c in cd.glob("*.ckpt"):
        m = CKPT_RE.match(c.name)
        if not m:
            continue
        ep = int(m.group(1))
        if ep > best_ep:
            best_ep, best = ep, c
    return best, best_ep


def pick_config(seed_dir: Path):
    pp = seed_dir / "pde_params_tsense"
    if not pp.is_dir():
        return None, None, -1
    best_cfg, best_ckpt_p, best_ep = None, None, -1
    for cfg in pp.glob("V*_config.json"):
        try:
            d = json.loads(cfg.read_text())
        except Exception:
            continue
        ckpt_dir = d.get("experiment_config", {}).get("checkpoint_dir")
        if not ckpt_dir:
            continue
        c, ep = best_ckpt(Path(ckpt_dir))
        if ep > best_ep:
            best_ep, best_cfg, best_ckpt_p = ep, cfg, c
    return best_cfg, best_ckpt_p, best_ep


def build_inventory(datasets, t_end_norm, noise_scale, sim_fn):
    expected = f"t{t_end_norm}_n{noise_scale}_{sim_fn}_fate_eval.csv"
    todos = []
    for ds in datasets:
        ds_root = PDP_DIR / "logs" / f"{ds}_ablation"
        if not ds_root.is_dir():
            print(f"[WARN] missing dataset root: {ds_root}", file=sys.stderr)
            continue
        for arm_dir in sorted(ds_root.glob("*")):
            if not arm_dir.is_dir():
                continue
            arm = arm_dir.name
            for s in (0, 1, 2):
                seed_dir = arm_dir / f"seed_{s}"
                if not seed_dir.is_dir():
                    continue
                cfg, ckpt, ep = pick_config(seed_dir)
                name = f"{ds}_ablation__{arm}__seed_{s}"
                out = RES_ROOT / name / expected
                todos.append({
                    "dataset": ds, "arm": arm, "seed": s, "name": name,
                    "config": cfg, "ckpt": ckpt, "epoch": ep, "output": out,
                })
    return todos


def make_sim_fn(args):
    if args.sim_fn == "ode":
        return fate_eval.make_pseudodynamics_sim_fn(
            t_start_norm=0.0, t_end_norm=args.t_end_norm,
        )
    if args.sim_fn == "sde":
        return fate_eval.make_pseudodynamics_sde_sim_fn(
            t_start_norm=0.0, t_end_norm=args.t_end_norm,
            n_steps=args.n_steps, noise_scale=args.noise_scale,
        )
    if args.sim_fn == "sb":
        return fate_eval.make_pseudodynamics_sb_sim_fn(
            t_start_norm=0.0, t_end_norm=args.t_end_norm,
            n_steps=args.n_steps, noise_scale=args.noise_scale,
        )
    raise ValueError(f"unknown sim_fn: {args.sim_fn}")


def main():
    p = argparse.ArgumentParser(
        description="Sweep fate-accuracy eval across Klein ablation checkpoints "
                    "(single-process, KNN fit once).",
    )
    p.add_argument("--datasets", nargs="+", default=["klein_OT", "klein_nonOT"])
    p.add_argument("--sim_fn", default="ode", choices=["ode", "sde", "sb"])
    p.add_argument("--t_end_norm", type=float, default=0.2)
    p.add_argument("--n_steps", type=int, default=200)
    p.add_argument("--noise_scale", type=float, default=1.0)
    p.add_argument("--n_sims", type=int, default=100)
    p.add_argument("--k", type=int, default=15,
                   help="KNN neighbours (default 15, matches 03_eval_pdp_fate.py default).")
    p.add_argument("--celltype_col", default="Annotation")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if the output CSV already exists.")
    p.add_argument("--skip_per_cell", action="store_true",
                   help="Skip writing the per-cell F_hat CSV (the large one).")
    p.add_argument("--log_dir", default=None,
                   help="Override sweep log directory (default: logs/eval_klein/fate_sweep_<ts>).")
    args = p.parse_args()

    log_dir = Path(args.log_dir) if args.log_dir else (
        PDP_DIR / "logs" / "eval_klein"
        / f"fate_sweep_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "_summary.tsv"

    print(f"[SWEEP] start={time.strftime('%F %T')}  device={args.device}")
    print(f"[SWEEP] datasets={args.datasets}  sim_fn={args.sim_fn}  "
          f"t_end={args.t_end_norm}  noise={args.noise_scale}  k={args.k}")
    print(f"[SWEEP] log_dir={log_dir}")
    print()

    # ── Inventory ─────────────────────────────────────────────────────────
    todos = build_inventory(args.datasets, args.t_end_norm,
                            args.noise_scale, args.sim_fn)
    if not todos:
        print("[SWEEP] No (dataset, arm, seed) entries found; aborting.")
        sys.exit(1)
    print(f"[SWEEP] {len(todos)} (dataset, arm, seed) entries")

    # Probe the first valid config so we know which cellstate_key / n_dims
    # to use when hoisting x_ref. All ablation arms within a dataset share
    # these (they only vary the λ weights); we *verify* per arm inside the
    # loop and skip any arm that disagrees.
    first_cfg = next((t["config"] for t in todos if t["config"] is not None), None)
    if first_cfg is None:
        print("[SWEEP] No valid configs in inventory; aborting.")
        sys.exit(1)
    cfg0 = pdp.ExperimentConfig(str(first_cfg))
    cfg0.from_json(str(first_cfg),
                   main_dir="/rds/user/wz369/hpc-work/pseudodynamics_plus/")
    cellstate_key = cfg0.dataset_config.get("cellstate_key", None)
    n_dims = cfg0.dataset_config["n_dimension"]
    if cellstate_key is None:
        print(f"[SWEEP] first config {first_cfg} has no cellstate_key; aborting.")
        sys.exit(1)
    print(f"[SWEEP] hoisted cellstate_key={cellstate_key}  n_dims={n_dims}")

    # ── Load adata + F_obs ONCE ───────────────────────────────────────────
    print("[SWEEP] loading data/klein_addpop.h5ad ...")
    adata = sc.read_h5ad("data/klein_addpop.h5ad")
    F_obs = pd.read_csv("data/klein/F_obs.csv", index_col=0)
    F_obs = F_obs[F_obs.index.astype(str).isin(adata.obs_names)]

    # Start cells from intersection of F_obs.index and adata.obs_names
    fobs_idx_str = F_obs.index.astype(str)
    valid_mask = adata.obs_names.astype(str).isin(fobs_idx_str)
    start_cells = adata[valid_mask].obsm[cellstate_key][:, :n_dims]
    start_cell_ids = list(adata[valid_mask].obs_names)

    # Reference set for KNN
    adata_train = adata[adata.obs.Well != 2]
    x_ref = adata_train.obsm[cellstate_key][:, :n_dims].astype(np.float32)
    y_ref = adata_train.obs[args.celltype_col].values.astype(str)

    # ── Fit KNN ONCE ──────────────────────────────────────────────────────
    if args.k > len(x_ref):
        print(f"[SWEEP] --k={args.k} exceeds n_ref={len(x_ref)}; aborting.",
              file=sys.stderr)
        sys.exit(1)
    print(f"[SWEEP] fitting KNN (k={args.k}, n_ref={len(y_ref)}) ...")
    t_knn = time.time()
    knn_pre = KNeighborsClassifier(n_neighbors=args.k, metric="euclidean")
    knn_pre.fit(x_ref, y_ref)
    print(f"[SWEEP] KNN fitted in {time.time() - t_knn:.2f}s "
          f"(n_start_cells={len(start_cells)})")

    # ── Loop over models ──────────────────────────────────────────────────
    with open(summary_path, "w") as fh:
        fh.write("status\tdataset\tarm\tseed\tepoch\telapsed_s\toutput\n")

    n_total = n_done = n_skip = n_fail = n_nockpt = n_mismatch = 0
    for i, t in enumerate(todos, 1):
        n_total += 1
        tag = f"{t['dataset']}/{t['arm']}/seed_{t['seed']}"
        out_path = t["output"]

        def log_row(status, elapsed=0, out=None):
            with open(summary_path, "a") as fh:
                fh.write(f"{status}\t{t['dataset']}\t{t['arm']}\t"
                         f"{t['seed']}\t{t['epoch']}\t{elapsed}\t"
                         f"{out if out is not None else out_path}\n")

        if t["config"] is None or t["ckpt"] is None:
            print(f"[{i}/{len(todos)}] SKIP (no ckpt) {tag}")
            n_nockpt += 1
            log_row("NOCKPT")
            continue

        if out_path.exists() and not args.force:
            print(f"[{i}/{len(todos)}] DONE (cached) {tag}  ep={t['epoch']}")
            n_skip += 1
            log_row("CACHED")
            continue

        print(f"[{i}/{len(todos)}] RUN  {tag}  ep={t['epoch']}  "
              f"cfg={Path(t['config']).name}")
        t0 = time.time()
        try:
            config = pdp.ExperimentConfig(str(t["config"]))
            config.from_json(str(t["config"]),
                             main_dir="/rds/user/wz369/hpc-work/pseudodynamics_plus/")

            # Verify cellstate consistency with the hoisted KNN. If an arm
            # disagrees we cannot reuse the pre-fit KNN safely; skip it.
            ck = config.dataset_config.get("cellstate_key", None)
            nd = config.dataset_config["n_dimension"]
            if ck != cellstate_key or nd != n_dims:
                print(f"       MISMATCH cellstate {ck}/{nd} vs "
                      f"hoisted {cellstate_key}/{n_dims}; skipping")
                n_mismatch += 1
                log_row("MISMATCH", elapsed=int(time.time() - t0))
                continue

            # Use the explicit ckpt picked by the inventory step (highest
            # epoch), not config.find_lastest_ckpt(), so the loaded model
            # matches the epoch we logged.
            pde_model = pdp.models.pde_params.load_from_checkpoint(
                str(t["ckpt"])
            ).to(args.device)
            sim_fn = make_sim_fn(args)

            result = fate_eval.run_fate_evaluation(
                start_cells, start_cell_ids, pde_model, sim_fn,
                F_obs, x_ref, y_ref,
                cell_types=None,
                n_sims=args.n_sims,
                k=args.k,
                device=args.device,
                knn=knn_pre,        # ← pre-fit KNN, no refit
            )

            out_path.parent.mkdir(parents=True, exist_ok=True)
            df_summary = pd.DataFrame([{
                "accuracy":      result["accuracy"],
                "pearson_r":     result["pearson_r"],
                "pearson_p":     result["pearson_p"],
                "n_start_cells": result["n_start_cells"],
                "n_sims":        args.n_sims,
                "k_nn":          args.k,
                "obsm_key":      cellstate_key,
                "n_dims":        n_dims,
                "t_end_norm":    args.t_end_norm,
                "noise_scale":   args.noise_scale,
                "sim_fn":        args.sim_fn,
                "n_steps":       args.n_steps,
                "model_dir":     str(t["config"]),
            }])
            df_summary.to_csv(out_path, index=False)

            F_hat = result["F_hat"]
            F_obs_al = result["F_obs_aligned"]
            fate_tbl = pd.DataFrame({
                "F_obs_mean": F_obs_al.mean(),
                "F_hat_mean": F_hat.mean(),
            })
            fate_tbl.to_csv(
                str(out_path).replace(".csv", "_celltype_fractions.csv")
            )
            if not args.skip_per_cell:
                F_hat.to_csv(str(out_path).replace(".csv", "_F_hat.csv"))

            elapsed = int(time.time() - t0)
            print(f"       ok  acc={result['accuracy']:.4f}  "
                  f"r={result['pearson_r']:.4f}  elapsed={elapsed}s")
            log_row("OK", elapsed=elapsed)
            n_done += 1

            # Free GPU memory between models.
            del pde_model, sim_fn, result, F_hat, F_obs_al
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            elapsed = int(time.time() - t0)
            err_log = log_dir / f"{t['dataset']}__{t['arm']}__seed_{t['seed']}.err"
            with open(err_log, "w") as fh:
                fh.write(f"{type(e).__name__}: {e}\n\n")
                traceback.print_exc(file=fh)
            print(f"       FAIL {type(e).__name__}: {e}  "
                  f"elapsed={elapsed}s  log={err_log}")
            log_row("FAIL", elapsed=elapsed, out=err_log)
            n_fail += 1

    print()
    print(f"[SWEEP] end={time.strftime('%F %T')}")
    print(f"[SWEEP] total={n_total}  ok={n_done}  cached={n_skip}  "
          f"nockpt={n_nockpt}  mismatch={n_mismatch}  fail={n_fail}")
    print(f"[SWEEP] summary → {summary_path}")


if __name__ == "__main__":
    main()
