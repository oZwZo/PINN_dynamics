"""
DeepRUOTv2 Combined Evaluation Script
======================================
Evaluates a trained DeepRUOTv2 model on held-out test cells (Well == 2).
Deterministic ODE propagation via f_net.v_net gives fate accuracy,
Pearson r, and W2 in one run.

DeepRUOT's training CSV maps timepoints to integer `samples`:
    tp2 -> samples=0   tp4 -> samples=1   tp6 -> samples=2
The ODE integrates in this same `samples` time domain, so:
    Fate : samples 0 -> 2  (tp2 -> tp6)
    W2   : samples 1 -> 2  (tp4 -> tp6)

Outputs
-------
  - eval_combined.csv
  - F_hat_ode.csv
  - w2_per_clone_ode.csv

Usage
-----
python scripts/DeepRUOT/03_evaluate.py \
    --model_dir /rds/user/wz369/hpc-work/DeepRUOTv2/results/klein_pca30 \
    --data_path data/klein/klein_addpop.h5ad \
    --obsm_key X_pca --n_dims 30 --device cuda:0
"""

import os
import sys
import argparse
import logging

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import yaml

# DeepRUOTv2 package for FNet
sys.path.insert(0, "/rds/user/wz369/hpc-work/DeepRUOTv2")
from DeepRUOT.models import FNet

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import fate_eval_pipeline as fate_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def make_deepruot_ode_sim_fn(t_start: float, t_end: float,
                             method: str = "dopri5",
                             atol: float = 1e-5, rtol: float = 1e-5):
    """Deterministic ODE simulation for DeepRUOT via f_net.v_net.

    model     : FNet     — only v_net is used
    returns   : ndarray (n_cells, n_dims)
    """
    from torchdiffeq import odeint

    def _simulate(start_cells, model, n_sims, device):
        f_net = model
        x0 = torch.tensor(start_cells, dtype=torch.float32, device=device)
        t_span = torch.tensor([t_start, t_end], dtype=torch.float32, device=device)
        f_net.eval()
        with torch.no_grad():
            traj = odeint(
                lambda t, z: f_net.v_net(t, z),
                x0, t_span, method=method, atol=atol, rtol=rtol,
            )
        return traj[-1].cpu().numpy()

    return _simulate


def parse_args():
    p = argparse.ArgumentParser(
        description="Combined evaluation for DeepRUOTv2 (fate accuracy + W2)"
    )
    p.add_argument("--model_dir", required=True,
                   help="Result dir holding model_final (or model_result) and params.yml")
    p.add_argument("--data_path", default="data/klein/klein_addpop.h5ad",
                   help="Path to .h5ad file (for barcodes, clones, obsm)")
    p.add_argument("--config", default=None,
                   help="YAML config with model/data blocks. Default: {model_dir}/params.yml")
    p.add_argument("--F_obs_path", default="data/klein/F_obs.csv")
    p.add_argument("--clone_proportions", default="data/klein/clone_proportions.csv")
    p.add_argument("--celltype_col", default="Annotation")
    p.add_argument("--obsm_key", default="X_pca",
                   help="adata.obsm key matching the training embedding (X_pca, DM_EigenVectors, ...)")
    p.add_argument("--n_dims", type=int, default=None,
                   help="Number of embedding dims. Default: data.dim from the config.")
    p.add_argument("--checkpoint", default=None,
                   help="Explicit FNet checkpoint path. Default: model_final or model_result.")
    p.add_argument("--k", type=int, default=15, help="KNN neighbors for fate")
    p.add_argument("--ode_method", default="dopri5")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default=None,
                   help="Output directory (default: model_dir)")
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_checkpoint(args):
    if args.checkpoint:
        return args.checkpoint
    for name in ("model_final", "model_result"):
        cand = os.path.join(args.model_dir, name)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"No checkpoint found in {args.model_dir} (looked for model_final, model_result)"
    )


def main():
    args = parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    log.info(f"Device: {device}")

    output_dir = args.output_dir or args.model_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── Config ──
    config_path = args.config or os.path.join(args.model_dir, "params.yml")
    config = load_config(config_path)
    model_cfg = config["model"]
    n_dims = args.n_dims if args.n_dims is not None else int(config["data"]["dim"])
    log.info(f"Config: {config_path}  in_out_dim={model_cfg['in_out_dim']}  n_dims={n_dims}")

    # ── Model ──
    f_net = FNet(
        in_out_dim=model_cfg["in_out_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        n_hiddens=model_cfg["n_hiddens"],
        activation=model_cfg["activation"],
    ).to(device)
    ckpt_path = resolve_checkpoint(args)
    f_net.load_state_dict(torch.load(ckpt_path, map_location=device))
    f_net.eval()
    log.info(f"Loaded FNet from {ckpt_path}")

    # ── Data ──
    log.info(f"Loading adata: {args.data_path}")
    adata = sc.read_h5ad(args.data_path)

    F_obs = pd.read_csv(args.F_obs_path, index_col=0)
    F_obs.index = F_obs.index.astype(str)
    F_obs = F_obs[F_obs.index.isin(adata.obs_names.astype(str))]

    clone_proportions = pd.read_csv(args.clone_proportions, index_col=0)

    # ═══════════════════════════════════════════════════════════════════════
    # FATE EVAL — start cells matched to F_obs
    # ═══════════════════════════════════════════════════════════════════════
    fobs_idx = F_obs.index.astype(str)
    valid_mask = adata.obs_names.astype(str).isin(fobs_idx)
    start_cells_fate = (
        adata[valid_mask].obsm[args.obsm_key][:, :n_dims].astype(np.float32)
    )
    start_barcodes = list(adata[valid_mask].obs_names.astype(str))
    log.info(f"Fate start cells: {len(start_cells_fate)} (matched from F_obs)")

    # KNN reference: training cells (Well != 2)
    adata_train = adata[adata.obs.Well != 2]
    x_ref = adata_train.obsm[args.obsm_key][:, :n_dims].astype(np.float32)
    y_ref = adata_train.obs[args.celltype_col].values.astype(str)

    # ═══════════════════════════════════════════════════════════════════════
    # W2 EVAL — clone-filtered test cells (Well == 2)
    # ═══════════════════════════════════════════════════════════════════════
    test_ad = adata[adata.obs.Well == 2]
    w2_clone_mask = test_ad.obs.clones.isin(clone_proportions.index)
    w2_test_ad = test_ad[w2_clone_mask]

    src_ad_w2 = w2_test_ad[w2_test_ad.obs.timepoint_tx_days == 4]
    tgt_ad_w2 = w2_test_ad[w2_test_ad.obs.timepoint_tx_days == 6]

    src_cells_w2 = src_ad_w2.obsm[args.obsm_key][:, :n_dims].astype(np.float32)
    tgt_cells_w2 = tgt_ad_w2.obsm[args.obsm_key][:, :n_dims].astype(np.float32)
    src_clones_w2 = src_ad_w2.obs.clones.values
    tgt_clones_w2 = tgt_ad_w2.obs.clones.values
    log.info(f"W2 src (tp4): {len(src_cells_w2)}  tgt (tp6): {len(tgt_cells_w2)}")

    # ═══════════════════════════════════════════════════════════════════════
    # EVALUATE (ODE — deterministic)
    # DeepRUOT uses samples {0,1,2} for tp {2,4,6}:
    #   fate  tp2 -> tp6  ==  samples 0 -> 2
    #   W2    tp4 -> tp6  ==  samples 1 -> 2
    # ═══════════════════════════════════════════════════════════════════════
    mode = "ode"
    log.info(f"\n{'='*60}\nMode: {mode.upper()}")

    # ── Fate accuracy ──
    sim_fn_fate = make_deepruot_ode_sim_fn(
        t_start=0.0, t_end=2.0, method=args.ode_method,
    )
    fate_result = fate_eval.run_fate_evaluation(
        start_cells=start_cells_fate,
        start_cell_ids=start_barcodes,
        model=f_net,
        simulation_func=sim_fn_fate,
        F_obs=F_obs,
        x_ref=x_ref, y_ref=y_ref,
        n_sims=1,
        k=args.k,
        device=device,
    )
    log.info(f"  Fate accuracy: {fate_result['accuracy']:.4f}")
    log.info(f"  Fate Pearson r: {fate_result['pearson_r']:.4f}")
    fate_result["F_hat"].to_csv(os.path.join(output_dir, f"F_hat_{mode}.csv"))

    # ── Population W2 ──
    sim_fn_w2 = make_deepruot_ode_sim_fn(
        t_start=1.0, t_end=2.0, method=args.ode_method,
    )
    w2_result = fate_eval.compute_w2_population(
        src_cells=src_cells_w2,
        target_cells=tgt_cells_w2,
        model=f_net,
        sim_fn=sim_fn_w2,
        n_sims=1,
        device=device,
        scaler=None,
    )
    log.info(f"  W2 scaled: {w2_result['w2_scaled']:.4f}")
    log.info(f"  W2 raw:    {w2_result['w2_raw']:.4f}")

    # ── Per-clone W2 ──
    df_clone_w2 = fate_eval.compute_w2_per_clone(
        src_cells=src_cells_w2,
        src_clone_ids=src_clones_w2,
        target_cells=tgt_cells_w2,
        target_clone_ids=tgt_clones_w2,
        model=f_net,
        sim_fn=sim_fn_w2,
        n_sims=1,
        device=device,
        scaler=None,
    )
    df_clone_w2.to_csv(os.path.join(output_dir, f"w2_per_clone_{mode}.csv"), index=False)
    log.info(f"  Per-clone W2: {len(df_clone_w2)} clones  "
             f"mean_scaled={df_clone_w2['w2_scaled'].mean():.4f}  "
             f"mean_raw={df_clone_w2['w2_raw'].mean():.4f}")
    log.info("  Per-clone W2 table:\n" + df_clone_w2.to_string(index=False))

    # ── Per-clone fate ──
    # Aggregate F_hat / F_obs_aligned by adata.obs.clones, then compute per-clone
    # Pearson r across cell-type fractions, argmax agreement, and clone size.
    from scipy.stats import pearsonr

    F_hat_df = fate_result["F_hat"].copy()
    F_obs_aln = fate_result["F_obs_aligned"].copy()

    barcode_to_clone = adata.obs["clones"].astype(str).to_dict()
    F_hat_df["_clone"] = F_hat_df.index.astype(str).map(barcode_to_clone)
    F_obs_aln["_clone"] = F_obs_aln.index.astype(str).map(barcode_to_clone)

    valid_clone = F_hat_df["_clone"].notna() & (F_hat_df["_clone"] != "nan")
    F_hat_df = F_hat_df[valid_clone]
    F_obs_aln = F_obs_aln[valid_clone]

    fate_cols = [c for c in F_hat_df.columns if c != "_clone"]
    F_hat_mean = F_hat_df.groupby("_clone")[fate_cols].mean()
    F_obs_mean = F_obs_aln.groupby("_clone")[fate_cols].mean()
    clone_sizes = F_hat_df.groupby("_clone").size()

    clone_rows = []
    for clone in F_hat_mean.index:
        h = F_hat_mean.loc[clone].values.astype(float)
        o = F_obs_mean.loc[clone].values.astype(float)
        try:
            r, _ = pearsonr(h, o)
        except Exception:
            r = float("nan")
        clone_rows.append({
            "clone": clone,
            "pearson_r": r,
            "accuracy_match": int(int(np.argmax(h)) == int(np.argmax(o))),
            "clone_size": int(clone_sizes.loc[clone]),
        })
    df_fate_clone = pd.DataFrame(clone_rows).sort_values(
        "pearson_r", ascending=False, na_position="last"
    )
    df_fate_clone.to_csv(os.path.join(output_dir, f"fate_per_clone_{mode}.csv"), index=False)
    log.info(f"  Per-clone fate: {len(df_fate_clone)} clones  "
             f"mean_r={df_fate_clone['pearson_r'].mean():.4f}  "
             f"mean_acc={df_fate_clone['accuracy_match'].mean():.4f}")
    log.info("  Per-clone fate table:\n" + df_fate_clone.to_string(index=False))

    # ── Combined CSV ──
    df_combined = pd.DataFrame([{
        "sim_mode": mode,
        "accuracy": fate_result["accuracy"],
        "pearson_r": fate_result["pearson_r"],
        "w2_scaled": w2_result["w2_scaled"],
        "w2_raw": w2_result["w2_raw"],
        "n_start_cells": fate_result["n_start_cells"],
        "n_sims_fate": 1,
        "k": args.k,
    }])
    combined_path = os.path.join(output_dir, "eval_combined.csv")
    df_combined.to_csv(combined_path, index=False)
    log.info(f"\n{'='*60}\nCombined results → {combined_path}")
    log.info(f"\n{df_combined.to_string(index=False)}")
    log.info("Done.")


if __name__ == "__main__":
    main()
