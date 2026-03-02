#!/rds/user/wz369/hpc-work/LIBS/mamba/envs/mioflow/bin/python
"""
TrajectoryNet Evaluation Script
=================================
Evaluates a trained TrajectoryNet model on held-out test cells:
  1. Forward simulation via CNF integration
  2. W2 (Wasserstein-2) distance between simulated and observed test cells
  3. Fate bias: KNN-assigned fate fractions vs. observed test fractions

Evaluation tasks:
  - tp=2->4 (norm 0->1): Simulate from train cells at tp=0 -> compare to test at tp=1
  - tp=4->6 (norm 1->2): Simulate from test cells at tp=1 -> compare to test at tp=2

Usage
-----
python scripts/TrajectoryNet/03_evaluate.py \
    --data_dir logs/TrajectoryNet/pca30 \
    --model_dir logs/TrajectoryNet/pca30/model --gpu 0

python scripts/TrajectoryNet/03_evaluate.py \
    --data_dir logs/TrajectoryNet/dm10 \
    --model_dir logs/TrajectoryNet/dm10/model --gpu 0
"""

import os
import sys
import json
import argparse
import logging
import sklearn
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a trained TrajectoryNet model (W2 + fate bias)"
    )
    p.add_argument("--data_dir", required=True,
                   help="Dir with scaler.npz, test_cells.npz, train_meta.npz, config.json")
    p.add_argument("--model_dir", required=True,
                   help="Dir with checkpt.pth and train_args.json")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU device index (default: 0)")
    p.add_argument("--n_sims", type=int, default=10,
                   help="Simulation replicates (default: 10)")
    p.add_argument("--n_sim_cells", type=int, default=5000,
                   help="Cells to simulate per replicate (default: 5000)")
    p.add_argument("--output", default=None,
                   help="Output CSV path (default: data_dir/eval_results.csv)")
    return p.parse_args()


def load_trajectorynet_model(model_dir, data_path, device):
    """Load a trained TrajectoryNet model from checkpoint + train_args.json."""
    from TrajectoryNet.parse import parser as tjn_parser
    from TrajectoryNet.train_misc import (
        build_model_tabular,
        create_regularization_fns,
        set_cnf_options,
    )
    from TrajectoryNet import dataset

    # Load training args
    train_args_path = os.path.join(model_dir, "train_args.json")
    with open(train_args_path) as f:
        train_args = json.load(f)

    # Build TrajectoryNet args via its parser
    parser_args = [
        "--dataset", data_path,
        "--embedding_name", train_args["embedding_name"],
        "--max_dim", str(train_args["max_dim"]),
        "--dims", train_args["dims"],
        "--time_scale", str(train_args["time_scale"]),
        "--layer_type", train_args["layer_type"],
        "--nonlinearity", train_args["nonlinearity"],
        "--save", os.path.abspath(model_dir),
        "--gpu", str(device.index if device.type == "cuda" else 0),
    ]
    tjn_args = tjn_parser.parse_args(parser_args)

    # Load dataset to get timepoints and shape
    data = dataset.SCData.factory(tjn_args.dataset, tjn_args)
    tjn_args.data = data
    tjn_args.timepoints = data.get_unique_times()
    tjn_args.int_tps = (np.arange(max(tjn_args.timepoints) + 1) + 1.0) * tjn_args.time_scale

    log.info(f"  Timepoints: {tjn_args.timepoints}")
    log.info(f"  Integration times: {tjn_args.int_tps}")
    log.info(f"  Data shape: {data.get_shape()}")

    # Build model
    regularization_fns, _ = create_regularization_fns(tjn_args)
    model = build_model_tabular(tjn_args, data.get_shape()[0], regularization_fns)

    # Load checkpoint
    ckpt_path = os.path.join(model_dir, "checkpt.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    log.info(f"  Loaded model from {ckpt_path}")

    return model, tjn_args


def forward_simulate_cnf(model, z_start, int_tp_start, int_tp_end, device):
    """Forward-integrate cells through the CNF from int_tp_start to int_tp_end."""
    z = z_start.clone().detach().to(device)
    zero = torch.zeros(z.shape[0], 1).to(device)

    integration_times = torch.tensor([int_tp_start, int_tp_end]).float().to(device)

    # For SequentialFlow with single CNF block, use chain[0] directly
    cnf = model.chain[0]
    z_out, delta_logp = cnf(z, zero, integration_times=integration_times, reverse=False)

    return z_out.detach()


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

    # ── load TrajectoryNet model ──
    data_npz_path = os.path.join(args.data_dir, "klein_train.npz")
    model, tjn_args = load_trajectorynet_model(args.model_dir, data_npz_path, device)
    int_tps = tjn_args.int_tps
    log.info(f"  int_tps: {int_tps}")

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

        # source cells: train cells at t_start (first transition),
        # test cells at t_start (subsequent transitions)
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

        # Integration time indices for this transition
        int_tp_start = float(int_tps[int(t_start)])
        int_tp_end = float(int_tps[int(t_end)])

        eval_tasks.append({
            "name": f"tp{t_start:.0f}_to_{t_end:.0f}",
            "t_start": t_start,
            "t_end": t_end,
            "int_tp_start": int_tp_start,
            "int_tp_end": int_tp_end,
            "src_emb": src_emb,
            "src_label": src_label,
            "target_emb": target_emb,
            "target_cts": target_cts,
            "knn_train_emb": knn_train_emb,
            "knn_train_cts": knn_train_cts,
        })

    # ── run evaluations ──
    all_results = []
    last_fate_results = {}

    for task in eval_tasks:
        log.info(f"\n{'='*60}")
        log.info(f"Eval: {task['name']} ({task['src_label']} cells at t={task['t_start']} "
                 f"-> test cells at t={task['t_end']})")
        log.info(f"  Source cells:  {task['src_emb'].shape[0]}")
        log.info(f"  Target cells:  {task['target_emb'].shape[0]}")
        log.info(f"  Integration:   [{task['int_tp_start']:.2f} -> {task['int_tp_end']:.2f}]")

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

            # forward simulate through CNF
            with torch.no_grad():
                z_sim = forward_simulate_cnf(
                    model, src_torch,
                    task['int_tp_start'], task['int_tp_end'],
                    device,
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
