#!/rds/user/wz369/hpc-work/LIBS/mamba/envs/mioflow/bin/python
"""
TIGON Evaluation Script
========================
Evaluates a trained TIGON model on held-out test cells:
  1. Forward simulation via TorchDiffEqPack's odesolve
  2. W2 (Wasserstein-2) distance between simulated and observed test cells
  3. Fate bias: KNN-assigned fate fractions vs. observed test fractions

Evaluation tasks:
  - tp=2->4 (norm 0->1): Simulate from train cells at tp=2 -> compare to test at tp=4
  - tp=4->6 (norm 1->2): Simulate from test cells at tp=4 -> compare to test at tp=6

Usage
-----
python scripts/TIGON/03_evaluate.py \
    --data_dir logs/TIGON/pca \
    --checkpoint logs/TIGON/pca/model/ckpt.pth \
    --hidden_dim 64 --gpu 0

python scripts/TIGON/03_evaluate.py \
    --data_dir logs/TIGON/dm \
    --checkpoint logs/TIGON/dm/model/ckpt.pth \
    --hidden_dim 64 --gpu 0
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

sys.path.insert(0, '/rds/user/wz369/hpc-work/TIGON')

from TorchDiffEqPack import odesolve
from utility import UOT, initialize_weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a trained TIGON model (W2 + fate bias)"
    )
    p.add_argument("--data_dir", required=True,
                   help="Dir with scaler.npz, test_cells.npz, train_meta.npz, config.json")
    p.add_argument("--checkpoint", required=True,
                   help="Path to TIGON .pth checkpoint")
    p.add_argument("--hidden_dim", type=int, default=64,
                   help="Hidden dim (must match training, default: 64)")
    p.add_argument("--n_hiddens", type=int, default=4,
                   help="Number of hidden layers (must match training, default: 4)")
    p.add_argument("--activation", default="Tanh",
                   help="Activation function (must match training, default: Tanh)")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU device index (default: 0)")
    p.add_argument("--n_sims", type=int, default=10,
                   help="Simulation replicates (default: 10)")
    p.add_argument("--n_sim_cells", type=int, default=5000,
                   help="Cells to simulate per replicate (default: 5000)")
    p.add_argument("--output", default=None,
                   help="Output CSV path (default: data_dir/eval_results.csv)")
    return p.parse_args()


def forward_simulate(func, z_start, t_start, t_end, device):
    """Forward ODE integration from t_start to t_end using TorchDiffEqPack."""
    z = z_start.clone().detach().to(device).requires_grad_(True)
    g0 = torch.zeros(z.shape[0], 1, dtype=torch.float32, device=device)
    logp0 = torch.zeros(z.shape[0], 1, dtype=torch.float32, device=device)

    options = {
        'method': 'Dopri5',
        'h': None,
        'rtol': 1e-3,
        'atol': 1e-5,
        'print_neval': False,
        'neval_max': 1000000,
        'safety': None,
        't0': t_start,
        't1': t_end,
    }

    z_out, g_out, logp_out = odesolve(func, y0=(z, g0, logp0), options=options)
    return z_out.detach(), g_out.detach()


def compute_w2(x_sim, x_true):
    """Compute exact W2 distance using POT library."""
    import ot
    x_s = x_sim.cpu().numpy().astype(np.float64)
    x_t = x_true.cpu().numpy().astype(np.float64)
    n_s = x_s.shape[0]
    n_t = x_t.shape[0]
    w_a = np.ones(n_s) / n_s
    w_b = np.ones(n_t) / n_t
    M = ot.dist(x_s, x_t, metric='sqeuclidean')
    w2_sq = ot.emd2(w_a, w_b, M)
    return float(np.sqrt(max(w2_sq, 0.0)))


def fate_bias_accuracy(x_sim, x_train, y_train, x_test_final, y_test_final, k=15):
    """
    Assign simulated cells to cell types via KNN trained on train coords.
    Compare simulated fate fractions to observed test-cell fate fractions.
    """
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
        pearson_r=r,
        pearson_p=pval,
        simulated_fractions=sim_frac,
        observed_fractions=obs_frac,
        cell_types=cell_types,
    )


def main():
    args = parse_args()

    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")

    output_path = args.output or os.path.join(args.data_dir, "eval_results.csv")

    # ── load config ──
    config_path = os.path.join(args.data_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    in_out_dim = config["in_out_dim"]
    norm_tps = config["normalized_timepoints"]
    log.info(f"  Config: in_out_dim={in_out_dim}, norm_tps={norm_tps}")

    # ── load model ──
    func = UOT(in_out_dim=in_out_dim,
               hidden_dim=args.hidden_dim,
               n_hiddens=args.n_hiddens,
               activation=args.activation).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    func.load_state_dict(checkpoint['func_state_dict'])
    func.eval()
    log.info(f"  Loaded model from {args.checkpoint}")

    # ── load train metadata ──
    train_meta = np.load(os.path.join(args.data_dir, "train_meta.npz"),
                         allow_pickle=True)
    train_embs = train_meta["embeddings"]  # object array, one per tp
    train_cts = train_meta["celltypes"]    # object array, one per tp

    # ── load test cells ──
    test_data = np.load(os.path.join(args.data_dir, "test_cells.npz"),
                        allow_pickle=True)
    test_emb = test_data["embeddings"]       # (n_test, n_dims)
    test_tps = test_data["timepoints"]       # normalized
    test_cts = test_data["celltypes"]

    log.info(f"  Test cells: {test_emb.shape[0]}")
    log.info(f"  Test timepoints (norm): {np.unique(test_tps)}")

    # ── define evaluation tasks ──
    # Task 1: tp=0 -> tp=1 (start from train tp=0, compare to test tp=1)
    # Task 2: tp=1 -> tp=2 (start from test tp=1, compare to test tp=2)
    eval_tasks = []
    for i in range(len(norm_tps) - 1):
        t_start = norm_tps[i]
        t_end = norm_tps[i + 1]

        # source cells: train cells at t_start (for first transition),
        # test cells at t_start (for subsequent transitions)
        if i == 0:
            src_emb = train_embs[i].astype(np.float32)
            src_label = "train"
        else:
            mask = np.isclose(test_tps, t_start, atol=1e-3)
            src_emb = test_emb[mask].astype(np.float32)
            src_label = "test"

        # target: test cells at t_end
        target_mask = np.isclose(test_tps, t_end, atol=1e-3)
        target_emb = test_emb[target_mask].astype(np.float32)
        target_cts = test_cts[target_mask]

        # KNN reference: train cells at target timepoint
        tp_idx = i + 1
        knn_train_emb = train_embs[tp_idx].astype(np.float32)
        knn_train_cts = train_cts[tp_idx]

        eval_tasks.append({
            "name": f"tp{t_start:.0f}_to_{t_end:.0f}",
            "t_start": t_start,
            "t_end": t_end,
            "src_emb": src_emb,
            "src_label": src_label,
            "target_emb": target_emb,
            "target_cts": target_cts,
            "knn_train_emb": knn_train_emb,
            "knn_train_cts": knn_train_cts,
        })

    # ── run evaluations ──
    all_results = []
    last_fate_results = {}  # store last replicate's fate result per task

    for task in eval_tasks:
        log.info(f"\n{'='*60}")
        log.info(f"Eval: {task['name']} ({task['src_label']} cells at t={task['t_start']} "
                 f"-> test cells at t={task['t_end']})")
        log.info(f"  Source cells:  {task['src_emb'].shape[0]}")
        log.info(f"  Target cells:  {task['target_emb'].shape[0]}")

        target_torch = torch.tensor(task['target_emb'], dtype=torch.float32)

        for rep in range(args.n_sims):
            # subsample source cells
            n_src = task['src_emb'].shape[0]
            if n_src > args.n_sim_cells:
                idx = np.random.choice(n_src, args.n_sim_cells, replace=False)
                src_batch = task['src_emb'][idx]
            else:
                src_batch = task['src_emb']

            src_torch = torch.tensor(src_batch, dtype=torch.float32)

            # forward simulate
            with torch.no_grad():
                z_sim, g_sim = forward_simulate(
                    func, src_torch, task['t_start'], task['t_end'], device
                )

            z_sim_cpu = z_sim.cpu()

            # subsample target for W2 feasibility
            n_tgt = task['target_emb'].shape[0]
            if n_tgt > args.n_sim_cells:
                tgt_idx = np.random.choice(n_tgt, args.n_sim_cells, replace=False)
                tgt_sub = target_torch[tgt_idx]
            else:
                tgt_sub = target_torch

            # subsample simulated cells to match
            if z_sim_cpu.shape[0] > args.n_sim_cells:
                sim_idx = np.random.choice(z_sim_cpu.shape[0], args.n_sim_cells, replace=False)
                sim_sub = z_sim_cpu[sim_idx]
            else:
                sim_sub = z_sim_cpu

            # W2
            w2 = compute_w2(sim_sub, tgt_sub)

            # fate bias
            fb = fate_bias_accuracy(
                x_sim=z_sim_cpu.numpy(),
                x_train=task['knn_train_emb'],
                y_train=task['knn_train_cts'],
                x_test_final=task['target_emb'],
                y_test_final=task['target_cts'],
            )

            all_results.append({
                "task": task['name'],
                "replicate": rep + 1,
                "w2": w2,
                "pearson_r": fb['pearson_r'],
                "pearson_p": fb['pearson_p'],
            })
            last_fate_results[task['name']] = fb

            log.info(f"  rep {rep+1:2d}/{args.n_sims}  W2={w2:.4f}  "
                     f"pearson_r={fb['pearson_r']:.4f}")

    # ── save results ──
    df = pd.DataFrame(all_results)

    # add summary rows per task
    summary_rows = []
    for task_name in df['task'].unique():
        task_df = df[df['task'] == task_name]
        summary_rows.append({
            "task": task_name,
            "replicate": "mean",
            "w2": task_df['w2'].mean(),
            "pearson_r": task_df['pearson_r'].mean(),
            "pearson_p": task_df['pearson_p'].mean(),
        })
        summary_rows.append({
            "task": task_name,
            "replicate": "std",
            "w2": task_df['w2'].std(),
            "pearson_r": task_df['pearson_r'].std(),
            "pearson_p": task_df['pearson_p'].std(),
        })

    df_summary = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_summary.to_csv(output_path, index=False)
    log.info(f"\nResults saved -> {output_path}")

    # ── print summary ──
    log.info(f"\n{'='*60}")
    log.info("Summary:")
    for task_name in df['task'].unique():
        task_df = df[df['task'] == task_name]
        log.info(f"  {task_name}:")
        log.info(f"    W2:        {task_df['w2'].mean():.4f} +/- {task_df['w2'].std():.4f}")
        log.info(f"    Pearson r: {task_df['pearson_r'].mean():.4f} +/- {task_df['pearson_r'].std():.4f}")

    # ── save fate fractions from last replicate of each task ──
    fate_rows = []
    for task in eval_tasks:
        fb = last_fate_results.get(task['name'])
        if fb is not None:
            for ct in fb['cell_types']:
                fate_rows.append({
                    "task": task['name'],
                    "cell_type": ct,
                    "simulated_fraction": fb['simulated_fractions'][ct],
                    "observed_fraction": fb['observed_fractions'][ct],
                })

    if fate_rows:
        fate_csv = output_path.replace(".csv", "_fate_fractions.csv")
        df_fate = pd.DataFrame(fate_rows)
        df_fate.to_csv(fate_csv, index=False)
        log.info(f"Fate fractions -> {fate_csv}")

    log.info("Done.")


if __name__ == "__main__":
    main()
