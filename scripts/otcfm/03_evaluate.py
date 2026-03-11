"""
OT-CFM Evaluation Script
========================
Evaluates a trained OT-CFM model on held-out test cells:
  1. W2 (Wasserstein-2) distance between simulated and observed test cells
  2. Fate bias: KNN-assigned fate fractions vs. observed test fractions
  3. Per-cell fate accuracy using shared fate_eval_pipeline

Evaluation tasks:
  - tp=0->1: Simulate from train cells at tp=0 -> compare to test at tp=1
  - tp=1->2: Simulate from test cells at tp=1 -> compare to test at tp=2

Usage
-----
python scripts/otcfm/03_evaluate.py \
    --model_dir logs/otcfm/pca30/model --gpu 0

python scripts/otcfm/03_evaluate.py \
    --model_dir logs/otcfm/dm10/model --gpu 0 \
    --F_obs_path data/klein/F_obs.csv
"""

import os
import sys
import json
import argparse
import logging

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr

from torchcfm.models import MLP
from torchcfm.utils import torch_wrapper
from torchdyn.core import NeuralODE

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained OT-CFM model")
    p.add_argument("--model_dir", required=True,
                   help="Dir with ckpt.pt, config.json, scaler.npz, train_meta.npz, test_cells.npz")
    p.add_argument("--F_obs_path", default="data/klein/F_obs.csv",
                   help="Path to ground-truth fate proportions CSV")
    p.add_argument("--celltype_col", default="Annotation",
                   help="Cell type column name")
    p.add_argument("--n_sims", type=int, default=10,
                   help="Simulation replicates for W2 (default: 10)")
    p.add_argument("--n_sims_fate", type=int, default=100,
                   help="Simulations per cell for fate accuracy (deterministic, so ignored)")
    p.add_argument("--n_sim_cells", type=int, default=5000,
                   help="Max cells to simulate per replicate for W2")
    p.add_argument("--k", type=int, default=15,
                   help="KNN neighbors for fate assignment")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU device index")
    p.add_argument("--output", default=None,
                   help="Output CSV path (default: model_dir/eval_results.csv)")
    return p.parse_args()


def load_model(model_dir, device):
    """Load OT-CFM model from checkpoint."""
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    n_dims = config["n_dims"]
    width = config["width"]

    model = MLP(dim=n_dims, time_varying=True, w=width).to(device)

    ckpt_path = os.path.join(model_dir, "ckpt.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, config


def forward_simulate_ode(model, x_start, t_start, t_end, device, n_steps=100):
    """Deterministic ODE integration using torchdyn NeuralODE."""
    node = NeuralODE(torch_wrapper(model), solver="dopri5")
    x0 = torch.tensor(x_start, dtype=torch.float32).to(device)
    with torch.no_grad():
        traj = node.trajectory(x0, t_span=torch.linspace(t_start, t_end, n_steps, device=device))
    return traj[-1].cpu().numpy()  # (n_cells, n_dims)


def compute_w2(x_sim, x_true):
    """Compute exact W2 distance using POT library."""
    import ot
    x_s = np.asarray(x_sim, dtype=np.float64)
    x_t = np.asarray(x_true, dtype=np.float64)
    n_s, n_t = x_s.shape[0], x_t.shape[0]
    w_a = np.ones(n_s) / n_s
    w_b = np.ones(n_t) / n_t
    M = ot.dist(x_s, x_t, metric='sqeuclidean')
    w2_sq = ot.emd2(w_a, w_b, M)
    return float(np.sqrt(max(w2_sq, 0.0)))


def fate_bias_accuracy(x_sim, x_train, y_train, x_test_final, y_test_final, k=15):
    """KNN fate fraction comparison (population-level Pearson r)."""
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(x_train, y_train)

    sim_labels = knn.predict(x_sim)
    cell_types = sorted(np.unique(np.concatenate([y_train, y_test_final])))

    sim_frac = {ct: np.mean(sim_labels == ct) for ct in cell_types}
    obs_frac = {ct: np.mean(y_test_final == ct) for ct in cell_types}

    sim_vec = np.array([sim_frac[ct] for ct in cell_types])
    obs_vec = np.array([obs_frac[ct] for ct in cell_types])

    if sim_vec.std() < 1e-10 or obs_vec.std() < 1e-10:
        r, pval = 0.0, 1.0
    else:
        r, pval = pearsonr(sim_vec, obs_vec)

    return dict(
        pearson_r=r, pearson_p=pval,
        simulated_fractions=sim_frac, observed_fractions=obs_frac,
        cell_types=cell_types,
    )


def main():
    args = parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    output_path = args.output or os.path.join(args.model_dir, "eval_results.csv")

    # ── Load model and config ──
    model, config = load_model(args.model_dir, device)
    norm_tps = config["normalized_timepoints"]
    n_dims = config["n_dims"]
    log.info(f"Config: n_dims={n_dims}, norm_tps={norm_tps}")

    # ── Load train metadata ──
    train_meta = np.load(os.path.join(args.model_dir, "train_meta.npz"), allow_pickle=True)
    train_embs = train_meta["embeddings"]
    train_cts = train_meta["celltypes"]

    # ── Load test cells ──
    test_data = np.load(os.path.join(args.model_dir, "test_cells.npz"), allow_pickle=True)
    test_emb = test_data["embeddings"]
    test_tps = test_data["timepoints"]
    test_cts = test_data["celltypes"]
    test_barcodes = test_data["barcodes"] if "barcodes" in test_data else None

    log.info(f"Test cells: {test_emb.shape[0]}")

    # ── Define evaluation tasks (W2 + fate bias) ──
    eval_tasks = []
    for i in range(len(norm_tps) - 1):
        t_start = norm_tps[i]
        t_end = norm_tps[i + 1]

        if i == 0:
            src_emb = train_embs[i].astype(np.float32)
            src_label = "train"
        else:
            mask = np.isclose(test_tps, t_start, atol=1e-3)
            src_emb = test_emb[mask].astype(np.float32)
            src_label = "test"

        target_mask = np.isclose(test_tps, t_end, atol=1e-3)
        target_emb = test_emb[target_mask].astype(np.float32)
        target_cts = test_cts[target_mask]

        knn_train_emb = train_embs[i + 1].astype(np.float32)
        knn_train_cts = train_cts[i + 1]

        eval_tasks.append({
            "name": f"tp{t_start}_to_{t_end}",
            "t_start": float(t_start),
            "t_end": float(t_end),
            "src_emb": src_emb,
            "src_label": src_label,
            "target_emb": target_emb,
            "target_cts": target_cts,
            "knn_train_emb": knn_train_emb,
            "knn_train_cts": knn_train_cts,
        })

    # ── Run W2 + fate bias evaluations ──
    all_results = []
    last_fate_results = {}

    for task in eval_tasks:
        log.info(f"\n{'='*60}")
        log.info(f"Eval: {task['name']} ({task['src_label']} -> test)")
        log.info(f"  Source: {task['src_emb'].shape[0]}, Target: {task['target_emb'].shape[0]}")

        for rep in range(args.n_sims):
            # Subsample source
            n_src = task['src_emb'].shape[0]
            if n_src > args.n_sim_cells:
                idx = np.random.choice(n_src, args.n_sim_cells, replace=False)
                src_batch = task['src_emb'][idx]
            else:
                src_batch = task['src_emb']

            # Forward simulate (deterministic ODE)
            z_sim = forward_simulate_ode(
                model, src_batch, task['t_start'], task['t_end'], device
            )

            # Subsample for W2
            n_tgt = task['target_emb'].shape[0]
            tgt_sub = task['target_emb']
            if n_tgt > args.n_sim_cells:
                tgt_idx = np.random.choice(n_tgt, args.n_sim_cells, replace=False)
                tgt_sub = task['target_emb'][tgt_idx]

            sim_sub = z_sim
            if z_sim.shape[0] > args.n_sim_cells:
                sim_idx = np.random.choice(z_sim.shape[0], args.n_sim_cells, replace=False)
                sim_sub = z_sim[sim_idx]

            w2 = compute_w2(sim_sub, tgt_sub)

            # Fate bias
            fb = fate_bias_accuracy(
                x_sim=z_sim,
                x_train=task['knn_train_emb'],
                y_train=task['knn_train_cts'],
                x_test_final=task['target_emb'],
                y_test_final=task['target_cts'],
                k=args.k,
            )

            all_results.append({
                "task": task['name'], "replicate": rep + 1,
                "w2": w2, "pearson_r": fb['pearson_r'], "pearson_p": fb['pearson_p'],
            })
            last_fate_results[task['name']] = fb
            log.info(f"  rep {rep+1}/{args.n_sims}  W2={w2:.4f}  r={fb['pearson_r']:.4f}")

    # ── Per-cell fate accuracy (using shared pipeline) ──
    fate_result = None
    if os.path.exists(args.F_obs_path) and test_barcodes is not None:
        log.info(f"\n{'='*60}")
        log.info("Per-cell fate accuracy evaluation")
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from fate_eval_pipeline import run_fate_evaluation, make_otcfm_sim_fn

            F_obs = pd.read_csv(args.F_obs_path, index_col=0)
            F_obs.index = F_obs.index.astype(str)

            # Start cells = test cells at tp=0 (day 2)
            tp0_mask = np.isclose(test_tps, 0.0, atol=1e-3)
            start_cells = test_emb[tp0_mask].astype(np.float32)
            start_barcodes = test_barcodes[tp0_mask]

            # KNN reference = train cells at last timepoint
            last_tp_idx = len(norm_tps) - 1
            x_ref = train_embs[last_tp_idx].astype(np.float32)
            y_ref = train_cts[last_tp_idx].astype(str)

            # Simulate from tp=0 to tp=n_times-1
            n_times = len(norm_tps)
            sim_fn = make_otcfm_sim_fn(t_start=0.0, t_end=float(n_times - 1))

            fate_result = run_fate_evaluation(
                start_cells=start_cells,
                start_cell_ids=start_barcodes,
                model=model,
                simulation_func=sim_fn,
                F_obs=F_obs,
                x_ref=x_ref,
                y_ref=y_ref,
                n_sims=1,  # deterministic
                k=args.k,
                device=str(device),
            )
            log.info(f"Fate accuracy: {fate_result['accuracy']:.4f}")
            log.info(f"Fate Pearson r: {fate_result['pearson_r']:.4f}")

            # Save F_hat
            fate_result['F_hat'].to_csv(
                os.path.join(args.model_dir, "F_hat_per_cell.csv"))
        except Exception as e:
            log.warning(f"Per-cell fate accuracy failed: {e}")

    # ── Save results ──
    df = pd.DataFrame(all_results)

    summary_rows = []
    for task_name in df['task'].unique():
        task_df = df[df['task'] == task_name]
        summary_rows.append({
            "task": task_name, "replicate": "mean",
            "w2": task_df['w2'].mean(),
            "pearson_r": task_df['pearson_r'].mean(),
            "pearson_p": task_df['pearson_p'].mean(),
        })
        summary_rows.append({
            "task": task_name, "replicate": "std",
            "w2": task_df['w2'].std(),
            "pearson_r": task_df['pearson_r'].std(),
            "pearson_p": task_df['pearson_p'].std(),
        })

    # Add fate accuracy row if available
    if fate_result is not None:
        summary_rows.append({
            "task": "fate_accuracy",
            "replicate": "result",
            "w2": np.nan,
            "pearson_r": fate_result['pearson_r'],
            "pearson_p": fate_result['pearson_p'],
        })

    df_summary = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_summary.to_csv(output_path, index=False)
    log.info(f"\nResults saved -> {output_path}")

    # ── Print summary ──
    log.info(f"\n{'='*60}")
    log.info("Summary:")
    for task_name in df['task'].unique():
        task_df = df[df['task'] == task_name]
        log.info(f"  {task_name}:")
        log.info(f"    W2:        {task_df['w2'].mean():.4f} +/- {task_df['w2'].std():.4f}")
        log.info(f"    Pearson r: {task_df['pearson_r'].mean():.4f} +/- {task_df['pearson_r'].std():.4f}")
    if fate_result is not None:
        log.info(f"  Per-cell fate accuracy: {fate_result['accuracy']:.4f}")
        log.info(f"  Per-cell fate Pearson r: {fate_result['pearson_r']:.4f}")

    # ── Save fate fractions ──
    fate_rows = []
    for task_name, fb in last_fate_results.items():
        for ct in fb['cell_types']:
            fate_rows.append({
                "task": task_name, "cell_type": ct,
                "simulated_fraction": fb['simulated_fractions'][ct],
                "observed_fraction": fb['observed_fractions'][ct],
            })
    if fate_rows:
        fate_csv = output_path.replace(".csv", "_fate_fractions.csv")
        pd.DataFrame(fate_rows).to_csv(fate_csv, index=False)
        log.info(f"Fate fractions -> {fate_csv}")

    log.info("Done.")


if __name__ == "__main__":
    main()
