"""
SF2M Evaluation Script
========================
Evaluates a trained SF2M model on held-out test cells:
  1. W2 (Wasserstein-2) distance between simulated and observed test cells
  2. Fate bias: KNN-assigned fate fractions vs. observed test fractions
  3. Per-cell fate accuracy using shared fate_eval_pipeline

Supports three evaluation modes:
  - ode:  Deterministic (drift model only, NeuralODE with euler)
  - sde:  Stochastic (drift + score via torchsde)
  - both: Run both modes

Usage
-----
python scripts/sf2m/03_evaluate.py \
    --model_dir logs/sf2m/pca30/model --eval_mode both --gpu 0

python scripts/sf2m/03_evaluate.py \
    --model_dir logs/sf2m/dm10/model --eval_mode sde --gpu 0 \
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
import torchsde
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr

from torchcfm.models import MLP
from torchcfm.utils import torch_wrapper
from torchdyn.core import NeuralODE

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained SF2M model")
    p.add_argument("--model_dir", required=True,
                   help="Dir with ckpt.pt, config.json, scaler.npz, etc.")
    p.add_argument("--F_obs_path", default="data/klein/F_obs.csv",
                   help="Path to ground-truth fate proportions CSV")
    p.add_argument("--celltype_col", default="Annotation")
    p.add_argument("--eval_mode", choices=["ode", "sde", "both"], default="both",
                   help="Evaluation mode: ode, sde, or both")
    p.add_argument("--n_sims", type=int, default=10,
                   help="Replicates for W2 evaluation")
    p.add_argument("--n_sims_fate", type=int, default=100,
                   help="SDE simulations per cell for fate accuracy")
    p.add_argument("--n_sim_cells", type=int, default=5000,
                   help="Max cells per W2 replicate")
    p.add_argument("--sde_steps", type=int, default=400,
                   help="SDE integration steps")
    p.add_argument("--k", type=int, default=15,
                   help="KNN neighbors for fate assignment")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output", default=None)
    return p.parse_args()


class SDE(torch.nn.Module):
    """SDE wrapper for SF2M: drift + score for stochastic integration."""
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, drift, score, input_size, sigma=0.05):
        super().__init__()
        self.drift = drift
        self.score = score
        self.input_size = input_size
        self.sigma = sigma

    def f(self, t, y):
        y = y.view(-1, *self.input_size)
        if len(t.shape) == len(y.shape):
            x = torch.cat([y, t], 1)
        else:
            x = torch.cat([y, t.repeat(y.shape[0])[:, None]], 1)
        return self.drift(x).flatten(start_dim=1) + self.score(x).flatten(start_dim=1)

    def g(self, t, y):
        return torch.ones_like(y) * self.sigma


def load_models(model_dir, device):
    """Load SF2M drift + score models from checkpoint."""
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    n_dims = config["n_dims"]
    width = config["width"]
    sigma = config["sigma"]

    drift_model = MLP(dim=n_dims, time_varying=True, w=width).to(device)
    score_model = MLP(dim=n_dims, time_varying=True, w=width).to(device)

    ckpt_path = os.path.join(model_dir, "ckpt.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    drift_model.load_state_dict(ckpt["drift_model_state_dict"])
    score_model.load_state_dict(ckpt["score_model_state_dict"])
    drift_model.eval()
    score_model.eval()

    return drift_model, score_model, config


def forward_simulate_ode(drift_model, x_start, t_start, t_end, device, n_steps=100):
    """Deterministic ODE using drift model only."""
    node = NeuralODE(torch_wrapper(drift_model), solver="euler")
    x0 = torch.tensor(x_start, dtype=torch.float32).to(device)
    with torch.no_grad():
        traj = node.trajectory(x0, t_span=torch.linspace(t_start, t_end, n_steps, device=device))
    return traj[-1].cpu().numpy()  # (n_cells, n_dims)


def forward_simulate_sde(drift_model, score_model, x_start, t_start, t_end,
                         sigma, device, n_steps=400):
    """Stochastic SDE using drift + score."""
    n_dims = x_start.shape[1]
    sde = SDE(drift_model, score_model, input_size=(n_dims,), sigma=sigma)
    x0 = torch.tensor(x_start, dtype=torch.float32).to(device)
    with torch.no_grad():
        traj = torchsde.sdeint(
            sde, x0, ts=torch.linspace(t_start, t_end, n_steps, device=device),
        )
    return traj[-1].cpu().numpy()  # (n_cells, n_dims)


def compute_w2(x_sim, x_true):
    """Compute exact W2 via POT."""
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
    """KNN fate fraction comparison."""
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


def run_w2_fate_eval(mode_name, simulate_fn, eval_tasks, args):
    """Run W2 + fate bias evaluation for a given simulation function."""
    all_results = []
    last_fate_results = {}

    for task in eval_tasks:
        log.info(f"\n  [{mode_name}] {task['name']} ({task['src_label']} -> test)")
        log.info(f"    Source: {task['src_emb'].shape[0]}, Target: {task['target_emb'].shape[0]}")

        for rep in range(args.n_sims):
            n_src = task['src_emb'].shape[0]
            if n_src > args.n_sim_cells:
                idx = np.random.choice(n_src, args.n_sim_cells, replace=False)
                src_batch = task['src_emb'][idx]
            else:
                src_batch = task['src_emb']

            z_sim = simulate_fn(src_batch, task['t_start'], task['t_end'])

            # W2
            n_tgt = task['target_emb'].shape[0]
            tgt_sub = task['target_emb']
            if n_tgt > args.n_sim_cells:
                tgt_sub = task['target_emb'][
                    np.random.choice(n_tgt, args.n_sim_cells, replace=False)]

            sim_sub = z_sim
            if z_sim.shape[0] > args.n_sim_cells:
                sim_sub = z_sim[
                    np.random.choice(z_sim.shape[0], args.n_sim_cells, replace=False)]

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
                "mode": mode_name, "task": task['name'], "replicate": rep + 1,
                "w2": w2, "pearson_r": fb['pearson_r'], "pearson_p": fb['pearson_p'],
            })
            last_fate_results[task['name']] = fb
            log.info(f"    rep {rep+1}/{args.n_sims}  W2={w2:.4f}  r={fb['pearson_r']:.4f}")

    return all_results, last_fate_results


def main():
    args = parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    output_path = args.output or os.path.join(args.model_dir, "eval_results.csv")

    # ── Load models ──
    drift_model, score_model, config = load_models(args.model_dir, device)
    norm_tps = config["normalized_timepoints"]
    n_dims = config["n_dims"]
    sigma = config["sigma"]
    log.info(f"Config: n_dims={n_dims}, sigma={sigma}, norm_tps={norm_tps}")

    # ── Load data ──
    train_meta = np.load(os.path.join(args.model_dir, "train_meta.npz"), allow_pickle=True)
    train_embs = train_meta["embeddings"]
    train_cts = train_meta["celltypes"]

    test_data = np.load(os.path.join(args.model_dir, "test_cells.npz"), allow_pickle=True)
    test_emb = test_data["embeddings"]
    test_tps = test_data["timepoints"]
    test_cts = test_data["celltypes"]
    test_barcodes = test_data["barcodes"] if "barcodes" in test_data else None

    log.info(f"Test cells: {test_emb.shape[0]}")

    # ── Build eval tasks ──
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
            "t_start": float(t_start), "t_end": float(t_end),
            "src_emb": src_emb, "src_label": src_label,
            "target_emb": target_emb, "target_cts": target_cts,
            "knn_train_emb": knn_train_emb, "knn_train_cts": knn_train_cts,
        })

    # ── Run evaluations ──
    modes = []
    if args.eval_mode in ("ode", "both"):
        modes.append("ode")
    if args.eval_mode in ("sde", "both"):
        modes.append("sde")

    all_results = []
    all_fate_results = {}

    for mode in modes:
        log.info(f"\n{'='*60}")
        log.info(f"Mode: {mode.upper()}")

        if mode == "ode":
            def sim_fn(x, t0, t1):
                return forward_simulate_ode(drift_model, x, t0, t1, device)
        else:
            def sim_fn(x, t0, t1):
                return forward_simulate_sde(
                    drift_model, score_model, x, t0, t1,
                    sigma, device, n_steps=args.sde_steps)

        results, fate_results = run_w2_fate_eval(mode, sim_fn, eval_tasks, args)
        all_results.extend(results)
        all_fate_results[mode] = fate_results

    # ── Per-cell fate accuracy ──
    fate_acc_results = {}
    if os.path.exists(args.F_obs_path) and test_barcodes is not None:
        log.info(f"\n{'='*60}")
        log.info("Per-cell fate accuracy evaluation")
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from fate_eval_pipeline import run_fate_evaluation, make_sf2m_ode_sim_fn, make_sf2m_sde_sim_fn

            F_obs = pd.read_csv(args.F_obs_path, index_col=0)
            F_obs.index = F_obs.index.astype(str)

            tp0_mask = np.isclose(test_tps, 0.0, atol=1e-3)
            start_cells = test_emb[tp0_mask].astype(np.float32)
            start_barcodes = test_barcodes[tp0_mask]

            last_tp_idx = len(norm_tps) - 1
            x_ref = train_embs[last_tp_idx].astype(np.float32)
            y_ref = train_cts[last_tp_idx].astype(str)

            n_times = len(norm_tps)

            for mode in modes:
                log.info(f"  Fate accuracy ({mode.upper()}):")
                if mode == "ode":
                    sim_fn = make_sf2m_ode_sim_fn(t_start=0.0, t_end=float(n_times - 1))
                    n_sims_fate = 1
                else:
                    sim_fn = make_sf2m_sde_sim_fn(
                        t_start=0.0, t_end=float(n_times - 1),
                        sigma=sigma, n_steps=args.sde_steps)
                    n_sims_fate = args.n_sims_fate

                model_arg = (drift_model, score_model) if mode == "sde" else drift_model
                result = run_fate_evaluation(
                    start_cells=start_cells,
                    start_cell_ids=start_barcodes,
                    model=model_arg,
                    simulation_func=sim_fn,
                    F_obs=F_obs,
                    x_ref=x_ref, y_ref=y_ref,
                    n_sims=n_sims_fate,
                    k=args.k,
                    device=str(device),
                )
                fate_acc_results[mode] = result
                log.info(f"    Accuracy: {result['accuracy']:.4f}")
                log.info(f"    Pearson r: {result['pearson_r']:.4f}")

                result['F_hat'].to_csv(
                    os.path.join(args.model_dir, f"F_hat_per_cell_{mode}.csv"))

        except Exception as e:
            log.warning(f"Per-cell fate accuracy failed: {e}")

    # ── Save results ──
    df = pd.DataFrame(all_results)

    summary_rows = []
    for mode in df['mode'].unique():
        for task_name in df[df['mode'] == mode]['task'].unique():
            task_df = df[(df['mode'] == mode) & (df['task'] == task_name)]
            summary_rows.append({
                "mode": mode, "task": task_name, "replicate": "mean",
                "w2": task_df['w2'].mean(),
                "pearson_r": task_df['pearson_r'].mean(),
                "pearson_p": task_df['pearson_p'].mean(),
            })
            summary_rows.append({
                "mode": mode, "task": task_name, "replicate": "std",
                "w2": task_df['w2'].std(),
                "pearson_r": task_df['pearson_r'].std(),
                "pearson_p": task_df['pearson_p'].std(),
            })

    for mode, result in fate_acc_results.items():
        summary_rows.append({
            "mode": mode, "task": "fate_accuracy", "replicate": "result",
            "w2": np.nan,
            "pearson_r": result['pearson_r'],
            "pearson_p": result['pearson_p'],
        })

    df_summary = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_summary.to_csv(output_path, index=False)
    log.info(f"\nResults saved -> {output_path}")

    # ── Print summary ──
    log.info(f"\n{'='*60}")
    log.info("Summary:")
    for mode in df['mode'].unique():
        log.info(f"  [{mode.upper()}]")
        for task_name in df[df['mode'] == mode]['task'].unique():
            task_df = df[(df['mode'] == mode) & (df['task'] == task_name)]
            log.info(f"    {task_name}:")
            log.info(f"      W2:        {task_df['w2'].mean():.4f} +/- {task_df['w2'].std():.4f}")
            log.info(f"      Pearson r: {task_df['pearson_r'].mean():.4f} +/- {task_df['pearson_r'].std():.4f}")
        if mode in fate_acc_results:
            log.info(f"    Per-cell fate accuracy: {fate_acc_results[mode]['accuracy']:.4f}")
            log.info(f"    Per-cell fate Pearson r: {fate_acc_results[mode]['pearson_r']:.4f}")

    # ── Save fate fractions ──
    fate_rows = []
    for mode, fate_res in all_fate_results.items():
        for task_name, fb in fate_res.items():
            for ct in fb['cell_types']:
                fate_rows.append({
                    "mode": mode, "task": task_name, "cell_type": ct,
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
