"""
Evaluation script for pseudodynamics+ on the Klein (Weinreb 2020) dataset.

Metrics
-------
Fate prediction (on F_obs clonal cells, all at tp2, Well=0):
  - F_hat: KNN-based fate probability from simulated tp6 positions
  - acc_raw:    compute_accuracy(F_obs, F_hat)          — undiff predictions always wrong
  - acc_nodiff: compute_accuracy(F_obs, replace_undiff(F_hat)) — renormalise differentiated

W2 distance (two variants):
  - w2_v1: propagate n_seed train-tp2 cells → compare with test (Well==2) at tp4 and tp6
  - w2_v2: propagate test-tp4 cells → compare with test-tp6 (tp4→tp6 prediction quality)

Per-clone fate distributions are saved as CSV.

Usage
-----
  python scripts/klein_eval.py \\
      --config logs/klein_DM_10_lD1_lv1_lgNone/V0_base.json \\
      --device cuda:0 \\
      --out results/klein_DM_lD1_lv1_lgNone
"""

import sys, os, argparse, json
sys.path.insert(0, os.path.abspath("src"))

import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.metrics
import torch
from torchdiffeq import odeint
from torchcfm.optimal_transport import wasserstein
from sklearn.neighbors import KNeighborsClassifier

import pseudodynamics as pdp
from pseudodynamics import models

# ── paths (relative to project root) ────────────────────────────────────────
F_OBS_PATH = "/rds/user/wz369/hpc-work/PINN_dynamics/data/klein/F_obs.csv"


# ── fate helpers (from user spec) ───────────────────────────────────────────

def replace_undiff(F_hat):
    """
    Re-normalise F_hat by removing the 'undiff' column.
    Cells whose only predicted fate was undiff keep undiff=1; all others
    are renormalised so differentiated columns sum to 1.
    """
    F_hat_filt = F_hat.copy().drop("undiff", axis=1)
    F_hat_filt_norm = F_hat_filt.div(F_hat_filt.sum(1), axis=0).fillna(0)

    undiff = np.zeros(len(F_hat_filt_norm))
    replace_idx = np.where(F_hat_filt_norm.sum(1) == 0)
    undiff[replace_idx] = 1

    F_hat_filt_norm["undiff"] = undiff
    F_hat_filt_norm.index = F_hat.index
    return F_hat_filt_norm


def compute_accuracy(F_obs, F_hat):
    """Top-1 accuracy: compare argmax of F_obs vs argmax of F_hat.

    Rows in F_obs that are all-zero (undiff cells with no known fate) are
    always counted as incorrect (contribute 0 to accuracy).
    """
    y_true = F_obs.idxmax(1).copy()
    undiff_mask = F_obs.sum(axis=1) == 0
    y_true[undiff_mask] = "__undiff__"   # sentinel — never matches any prediction
    y_pred = F_hat.idxmax(1).tolist()
    return sklearn.metrics.accuracy_score(y_true, y_pred)


# ── model / data loading ─────────────────────────────────────────────────────

def load_model(config_path, device):
    config = pdp.ExperimentConfig(config=config_path)
    ckpt = config.find_lastest_ckpt()
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found for {config_path}")
    pde_model = models.pde_params.load_from_checkpoint(ckpt, map_location="cpu")
    pde_model.eval()
    return config, pde_model.to(device)


# ── ODE propagation ──────────────────────────────────────────────────────────

def propagate(pde_model, X_seed, t_eval_norm, device, tol=1e-5):
    """
    Integrate the velocity ODE.

    t_eval_norm : 1-D array in the normalised time domain
        (norm_time='min_minus' → [0, 2, 4] for timepoints [2, 4, 6])
    Network receives t * time_scale_factor internally.
    Returns ndarray (len(t_eval_norm), n_seed, n_dim).
    """
    tsf = pde_model.time_scale_factor

    def v_ode(t, state):
        t_in = torch.full((state.shape[0], 1),
                          t.item() * tsf).float().to(device)
        return pde_model.v(state, t_in)

    s0 = torch.from_numpy(X_seed).float().to(device)
    t_tensor = torch.tensor(t_eval_norm, dtype=torch.float32).to(device)

    with torch.no_grad():
        traj = odeint(v_ode, s0, t_tensor,
                      method="dopri5", atol=tol, rtol=tol)
    return traj.cpu().numpy()   # (T, n_seed, n_dim)


# ── KNN fate-probability matrix ──────────────────────────────────────────────

def compute_F_hat(pred_positions, ref_X, ref_labels, all_types, k=15):
    """
    For each predicted cell, compute weighted KNN fate probability vector.

    Returns
    -------
    F_hat : pd.DataFrame  shape (n_pred, len(all_types))
    """
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1)
    knn.fit(ref_X, ref_labels)

    # weighted probability matrix
    proba = knn.predict_proba(pred_positions)   # (n_pred, n_knn_classes)
    knn_classes = list(knn.classes_)

    # map to full all_types order (fill 0 for missing classes)
    F_hat = pd.DataFrame(0.0, index=range(len(pred_positions)), columns=all_types)
    for j, ct in enumerate(knn_classes):
        if ct in all_types:
            F_hat[ct] = proba[:, j]

    return F_hat


# ── fate evaluation ──────────────────────────────────────────────────────────

def eval_fate(config, pde_model, adata, device, k_knn=15, out_dir=None):
    """
    Step 1 – start cells: test cells (Well==2) at tp4.
    Step 2 – propagate tp4 → tp6 (norm time 2 → 4).
    Step 3 – compute F_hat via KNN from tp6 cells.
             F_obs is constructed as one-hot from the tp4 cells' own Annotation.
             Two accuracy variants + per-clone CSVs saved.
    """
    raw           = config.raw_args
    cellstate_key = raw["cellstate_key"]
    n_dim         = raw["n_dimension"]
    tp_col        = "timepoint_tx_days"
    well_col      = "Well"
    label_col     = "Annotation"

    # fate type ordering from F_obs file (10 differentiated + undiff)
    f_obs_ref   = pd.read_csv(F_OBS_PATH, index_col=0)
    fated_types = f_obs_ref.columns.tolist()      # 10 differentiated types
    all_types   = fated_types + ["undiff"]        # 11 types for F_hat

    # ── Step 1: start cells — test tp4 ──
    test_mask = adata.obs[well_col].astype(int) == 2
    start_ad  = adata[(test_mask) & (adata.obs[tp_col] == 4.0)]

    X_seed = start_ad.obsm[cellstate_key][:, :n_dim]
    print(f"  Fate start cells: {len(X_seed)} test cells at tp4")

    # F_obs: one-hot from Annotation of each tp4 test cell
    # columns = fated_types only (no undiff, matching F_obs format)
    f_obs_sub = pd.get_dummies(start_ad.obs[label_col])
    f_obs_sub = f_obs_sub.reindex(columns=fated_types, fill_value=0)

    # ── Step 2: propagate tp4 → tp6 (t=2 → t=4 in norm_time) ──
    tsf    = pde_model.time_scale_factor
    t_eval = np.array([2.0, 4.0]) / tsf
    traj   = propagate(pde_model, X_seed, t_eval, device)
    pred6  = traj[-1]                              # (n_start, n_dim)

    # ── reference cells at tp6 (all wells) ──
    ref_ad = adata[adata.obs[tp_col] == 6.0]
    X_ref6 = ref_ad.obsm[cellstate_key][:, :n_dim]
    y_ref6 = ref_ad.obs[label_col].values

    # ── Step 3: compute F_hat ──
    print("  Computing KNN fate probabilities...")
    F_hat = compute_F_hat(pred6, X_ref6, y_ref6, all_types, k=k_knn)
    F_hat.index = start_ad.obs_names

    # accuracy variant 1 — raw (undiff predictions always wrong)
    acc_raw = compute_accuracy(f_obs_sub, F_hat)

    # accuracy variant 2 — replace undiff
    F_hat_nd   = replace_undiff(F_hat)
    acc_nodiff = compute_accuracy(f_obs_sub, F_hat_nd)

    print(f"  acc_raw={acc_raw:.4f}  acc_nodiff={acc_nodiff:.4f}")

    # ── per-clone fate distributions ──
    clone_col = "clones"
    clone_ids = start_ad.obs[clone_col].values
    F_hat["clone"] = clone_ids

    F_hat_clone    = F_hat.groupby("clone")[all_types].mean()
    F_hat_nd_clone = replace_undiff(F_hat.drop("clone", axis=1))
    F_hat_nd_clone["clone"] = clone_ids
    F_hat_nd_clone = F_hat_nd_clone.groupby("clone")[all_types].mean()

    F_hat = F_hat.drop("clone", axis=1)    # restore

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        F_hat.to_csv(os.path.join(out_dir, "F_hat_cellwise.csv"))
        F_hat_clone.to_csv(os.path.join(out_dir, "F_hat_clone.csv"))
        F_hat_nd.to_csv(os.path.join(out_dir, "F_hat_nodiff_cellwise.csv"))
        F_hat_nd_clone.to_csv(os.path.join(out_dir, "F_hat_nodiff_clone.csv"))
        f_obs_sub.to_csv(os.path.join(out_dir, "F_obs_tp4.csv"))
        print(f"  Saved fate CSVs to {out_dir}")

    return {
        "acc_raw":       acc_raw,
        "acc_nodiff":    acc_nodiff,
        "n_start_cells": len(X_seed),
        "n_clones":      F_hat_clone.shape[0],
    }


# ── W2 evaluation ────────────────────────────────────────────────────────────

def _w2(pred_tensor, ref_array, reg=0.05):
    ref_t = torch.from_numpy(ref_array).float()
    result = wasserstein(pred_tensor.float(), ref_t,
                         power=2, method="sinkhorn", reg=reg)
    return float(result)


def eval_w2_v1(config, pde_model, adata, device, n_seed=5000, reg=0.05):
    """
    V1: propagate n_seed random train-tp2 cells → compare with test tp4 and tp6.
    Tests: does the model correctly predict the test distribution?
    """
    raw           = config.raw_args
    cellstate_key = raw["cellstate_key"]
    n_dim         = raw["n_dimension"]
    tp_col        = "timepoint_tx_days"
    well_col      = "Well"

    train_mask = adata.obs[well_col].astype(int) != 2
    test_mask  = ~train_mask

    seed_ad = adata[(train_mask) & (adata.obs[tp_col] == 2.0)]
    np.random.seed(42)
    idx    = np.random.choice(len(seed_ad), min(n_seed, len(seed_ad)), replace=False)
    X_seed = seed_ad.obsm[cellstate_key][idx, :n_dim]

    test4_X = adata[(test_mask) & (adata.obs[tp_col] == 4.0)].obsm[cellstate_key][:, :n_dim]
    test6_X = adata[(test_mask) & (adata.obs[tp_col] == 6.0)].obsm[cellstate_key][:, :n_dim]

    tsf    = pde_model.time_scale_factor
    t_eval = np.array([0.0, 2.0, 4.0]) / tsf
    traj   = propagate(pde_model, X_seed, t_eval, device)

    pred4 = torch.from_numpy(traj[1])
    pred6 = torch.from_numpy(traj[2])

    w2_4 = _w2(pred4, test4_X, reg)
    w2_6 = _w2(pred6, test6_X, reg)

    return {"w2_v1_tp4": w2_4, "w2_v1_tp6": w2_6,
            "w2_v1_mean": (w2_4 + w2_6) / 2}


def eval_w2_v2(config, pde_model, adata, device, reg=0.05):
    """
    V2: start from test-tp4 cells → simulate to tp6 → compare with test-tp6.
    Tests: does the velocity field correctly advance the test distribution one step?
    """
    raw           = config.raw_args
    cellstate_key = raw["cellstate_key"]
    n_dim         = raw["n_dimension"]
    tp_col        = "timepoint_tx_days"
    well_col      = "Well"

    test_mask = adata.obs[well_col].astype(int) == 2
    test4_ad  = adata[(test_mask) & (adata.obs[tp_col] == 4.0)]
    test6_X   = adata[(test_mask) & (adata.obs[tp_col] == 6.0)].obsm[cellstate_key][:, :n_dim]

    X_seed = test4_ad.obsm[cellstate_key][:, :n_dim]

    # integrate from t=2 → t=4 (norm_time units)
    tsf    = pde_model.time_scale_factor
    t_eval = np.array([2.0, 4.0]) / tsf
    traj   = propagate(pde_model, X_seed, t_eval, device)

    pred6 = torch.from_numpy(traj[-1])
    w2_6  = _w2(pred6, test6_X, reg)

    return {"w2_v2_tp6": w2_6}


# ── fate accuracy via shared pipeline ────────────────────────────────────────

# Path to the method-agnostic fate evaluation pipeline (sibling repo)
_PIPELINE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../PINN_dynamics/scripts"
)


def eval_fate_pipeline(
    config,
    pde_model,
    adata,
    device,
    k_knn: int = 15,
    tol: float = 1e-5,
    out_dir=None,
):
    """
    Per-cell fate accuracy using fate_eval_pipeline.run_fate_evaluation.

    Differences from eval_fate():
    - Start cells  : F_obs clone-traced cells (all at tp2, Well=0) rather
                     than test cells at tp4.
    - Ground truth : clone-traced F_obs proportions (F_obs.csv) rather than
                     one-hot cell-type annotations.
    - Simulation   : deterministic ODE tp2 → tp6 (norm_time 0 → 4).
    - KNN ref      : all tp6 cells (same as eval_fate).
    - Metrics      : accuracy (argmax match) + population Pearson r via
                     fate_eval_pipeline.compute_fate_accuracy.

    Returns
    -------
    dict with keys: accuracy, pearson_r, pearson_p, n_start_cells,
                    F_hat (DataFrame), F_obs_aligned (DataFrame).
    """
    import sys
    if _PIPELINE_DIR not in sys.path:
        sys.path.insert(0, _PIPELINE_DIR)
    from fate_eval_pipeline import run_fate_evaluation, make_pseudodynamics_sim_fn

    raw           = config.raw_args
    cellstate_key = raw["cellstate_key"]
    n_dim         = raw["n_dimension"]
    tp_col        = "timepoint_tx_days"
    label_col     = "Annotation"

    # ── Load F_obs ────────────────────────────────────────────────────────
    F_obs = pd.read_csv(F_OBS_PATH, index_col=0)
    F_obs.index = F_obs.index.astype(str)

    # ── Start cells: F_obs cells at tp2 ──────────────────────────────────
    tp2_ad   = adata[adata.obs[tp_col] == 2.0]
    tp2_ids  = tp2_ad.obs_names.astype(str)
    fobs_set = set(F_obs.index)
    valid    = [i for i, id_ in enumerate(tp2_ids) if id_ in fobs_set]

    if not valid:
        raise ValueError(
            f"No F_obs cells found at tp=2 in adata.\n"
            f"  F_obs sample: {list(F_obs.index[:5])}\n"
            f"  adata tp2 sample: {list(tp2_ids[:5])}"
        )

    start_ad    = tp2_ad[valid]
    start_cells = start_ad.obsm[cellstate_key][:, :n_dim].astype(np.float32)
    start_ids   = start_ad.obs_names.astype(str).tolist()
    print(f"  Fate start cells (F_obs ∩ tp2): {len(start_cells)}")

    # ── KNN reference: all tp6 cells ─────────────────────────────────────
    ref_ad = adata[adata.obs[tp_col] == 6.0]
    x_ref  = ref_ad.obsm[cellstate_key][:, :n_dim].astype(np.float32)
    y_ref  = ref_ad.obs[label_col].values.astype(str)

    # ── Simulation function: deterministic ODE tp2→tp6 ───────────────────
    # Klein norm_time='min_minus': tp2=0, tp4=2, tp6=4
    sim_fn = make_pseudodynamics_sim_fn(t_start_norm=0.0, t_end_norm=4.0, tol=tol)

    # ── Run evaluation ────────────────────────────────────────────────────
    result = run_fate_evaluation(
        start_cells     = start_cells,
        start_cell_ids  = start_ids,
        model           = pde_model,
        simulation_func = sim_fn,
        F_obs           = F_obs,
        x_ref           = x_ref,
        y_ref           = y_ref,
        cell_types      = sorted(F_obs.columns.tolist()),
        n_sims          = 1,   # deterministic ODE
        k               = k_knn,
        device          = str(device),
    )

    print(f"  acc={result['accuracy']:.4f}  pearson_r={result['pearson_r']:.4f}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        result["F_hat"].to_csv(os.path.join(out_dir, "F_hat_pipeline.csv"))
        result["F_obs_aligned"].to_csv(os.path.join(out_dir, "F_obs_pipeline.csv"))
        print(f"  Saved pipeline fate CSVs to {out_dir}")

    return {
        "acc_pipeline":      result["accuracy"],
        "pearson_r_pipeline": result["pearson_r"],
        "pearson_p_pipeline": result["pearson_p"],
        "n_start_cells_pipeline": result["n_start_cells"],
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pseudodynamics+ on Klein dataset")
    parser.add_argument("--config",   required=True,
                        help="Path to experiment config JSON")
    parser.add_argument("--device",   default="cpu",
                        help="cpu or cuda:N  (default: cpu)")
    parser.add_argument("--n_seed",   type=int, default=5000,
                        help="Train-tp2 seed cells for W2-v1 (default: 5000)")
    parser.add_argument("--knn",      type=int, default=15,
                        help="K for KNN fate assignment (default: 15)")
    parser.add_argument("--reg",      type=float, default=0.05,
                        help="Sinkhorn regularisation for W2 (default: 0.05)")
    parser.add_argument("--out",      default=None,
                        help="Output directory for CSVs and results JSON")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Config : {args.config}")
    print(f"Device : {args.device}")
    print(f"{'='*60}\n")

    config, pde_model = load_model(args.config, args.device)
    raw = config.raw_args

    print("Loading adata...")
    adata = sc.read_h5ad(f"data/{raw['dataset']}.h5ad")

    fate_out = os.path.join(args.out, "fate") if args.out else None

    print("\n── Fate prediction (annotation one-hot) ─────────────")
    fate_res = eval_fate(config, pde_model, adata, args.device,
                         k_knn=args.knn, out_dir=fate_out)

    print("\n── Fate prediction (clone-traced F_obs pipeline) ────")
    pipeline_out = os.path.join(args.out, "fate_pipeline") if args.out else None
    fate_pipeline_res = eval_fate_pipeline(config, pde_model, adata, args.device,
                                           k_knn=args.knn, out_dir=pipeline_out)

    print("\n── W2 v1 (train tp2 → test tp4/tp6) ────────────────")
    w2_v1 = eval_w2_v1(config, pde_model, adata, args.device,
                        n_seed=args.n_seed, reg=args.reg)
    for k, v in w2_v1.items():
        print(f"  {k}: {v:.4f}")

    print("\n── W2 v2 (test tp4 → test tp6) ─────────────────────")
    w2_v2 = eval_w2_v2(config, pde_model, adata, args.device, reg=args.reg)
    for k, v in w2_v2.items():
        print(f"  {k}: {v:.4f}")

    results = {
        "config":          args.config,
        "cellstate_key":   raw["cellstate_key"],
        "n_dimension":     raw["n_dimension"],
        "D_penalty":       raw.get("D_penalty"),
        "deltax_weight":   raw.get("deltax_weight"),
        "growth_weight":   raw.get("growth_weight"),
        "weight_intensity": raw.get("weight_intensity"),
        **fate_res,
        **fate_pipeline_res,
        **w2_v1,
        **w2_v2,
    }

    print("\n── Summary ──────────────────────────────────────────")
    for k, v in results.items():
        print(f"  {k:<22}: {v:.4f}" if isinstance(v, float) else
              f"  {k:<22}: {v}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        # save scalar results
        with open(os.path.join(args.out, "metrics.json"), "w") as f:
            json.dump(results, f, indent=4)
        # save as single-row CSV for easy aggregation
        pd.DataFrame([results]).to_csv(
            os.path.join(args.out, "metrics.csv"), index=False)
        print(f"\nSaved metrics to {args.out}/")


if __name__ == "__main__":
    main()
