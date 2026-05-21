"""
SF2M DM10 Diagnostic — Tests 1-3
=================================
Standalone diagnostic script. Does NOT modify any existing scripts.
Run inside singularity on a GPU node:

    singularity exec --nv /rds/user/wz369/hpc-work/containers/flow.sif python \
        scripts/sf2m/diag_sf2m_dm10.py --device cuda:0

Tests:
  1. Solver sweep: same trained model, different ODE solvers / step counts
  2. Identity reconstruction: integrate t→t (zero travel), measure numerical error
  3. Per-dimension W2 decomposition: which dims contribute most to the gap?
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="logs/sf2m/dm10/model_r1")
    p.add_argument("--data_path", default="data/klein_addpop.h5ad")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default=None,
                   help="Where to save results (default: model_dir/diagnostics/)")
    p.add_argument("--max_cells", type=int, default=2000,
                   help="Max cells per side for W2 (subsample for speed)")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    output_dir = args.output_dir or os.path.join(args.model_dir, "diagnostics")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load model ──
    from torchcfm.models import MLP
    from torchcfm.utils import torch_wrapper
    from torchdyn.core import NeuralODE

    config = json.load(open(os.path.join(args.model_dir, "config.json")))
    n_dims = config["n_dims"]
    width = config["width"]
    norm_tps = config["normalized_timepoints"]

    drift = MLP(dim=n_dims, time_varying=True, w=width).to(device)
    ckpt = torch.load(os.path.join(args.model_dir, "ckpt.pt"),
                      map_location=device, weights_only=False)
    drift.load_state_dict(ckpt["drift_model_state_dict"])
    drift.eval()
    print(f"Loaded drift model from {args.model_dir} (best_iter={ckpt.get('best_iter')})")

    # ── Load test cells ──
    test_data = np.load(os.path.join(args.model_dir, "test_cells.npz"), allow_pickle=True)
    test_emb = test_data["embeddings"]
    test_tps = test_data["timepoints"]

    # Source = tp1 (normalized_timepoints[1]), Target = tp_last
    tp1 = float(norm_tps[1])
    tp_last = float(norm_tps[-1])
    src_mask = np.isclose(test_tps, tp1, atol=0.1)
    tgt_mask = np.isclose(test_tps, tp_last, atol=0.1)

    src_cells_full = test_emb[src_mask].astype(np.float32)
    tgt_cells_full = test_emb[tgt_mask].astype(np.float32)
    print(f"Source cells (tp={tp1}): {len(src_cells_full)}")
    print(f"Target cells (tp={tp_last}): {len(tgt_cells_full)}")
    print(f"Source per-dim std: {src_cells_full.std(0).mean():.6f}")
    print(f"Target per-dim std: {tgt_cells_full.std(0).mean():.6f}")

    # Subsample for fast W2 (full cells used for ODE integration, subset for W2)
    rng = np.random.RandomState(42)
    mc = args.max_cells
    if len(src_cells_full) > mc:
        src_idx = rng.choice(len(src_cells_full), mc, replace=False)
    else:
        src_idx = np.arange(len(src_cells_full))
    if len(tgt_cells_full) > mc:
        tgt_idx = rng.choice(len(tgt_cells_full), mc, replace=False)
    else:
        tgt_idx = np.arange(len(tgt_cells_full))
    src_cells = src_cells_full[src_idx]
    tgt_cells = tgt_cells_full[tgt_idx]
    print(f"Subsampled for W2: src={len(src_cells)}, tgt={len(tgt_cells)} (max_cells={mc})")

    # ── Load adata scaler for raw-space W2 ──
    import scanpy as sc
    adata = sc.read_h5ad(args.data_path)
    dm_scaler = None
    if "DM_scaler" in adata.uns:
        dm_scaler = adata.uns["DM_scaler"]
        print(f"DM_scaler loaded: std mean = {np.array(dm_scaler['std'][:n_dims]).mean():.6f}")

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 1: Solver sweep
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 1: SOLVER SWEEP (same model, different ODE integrators)")
    print("=" * 70)

    import fate_eval_pipeline as fate_eval

    solver_results = []
    t_start_w2 = tp1
    t_end_w2 = tp_last

    x0_t = torch.tensor(src_cells, dtype=torch.float32).to(device)

    for solver in ["euler", "midpoint", "rk4", "dopri5"]:
        for n_steps in [100, 500, 1000]:
            node = NeuralODE(torch_wrapper(drift), solver=solver)
            with torch.no_grad():
                traj = node.trajectory(
                    x0_t,
                    t_span=torch.linspace(t_start_w2, t_end_w2, n_steps, device=device),
                )
            ep = traj[-1].cpu().numpy()

            # W2 in current space (raw DM)
            w2_scaled = fate_eval.compute_w2(ep, tgt_cells, device=device)

            # W2 in raw space (should be same since no standardize, but compute for completeness)
            w2_raw = w2_scaled
            if dm_scaler is not None and config.get("standardize", False):
                ep_raw = fate_eval.inverse_standardize(ep, dm_scaler)
                tgt_raw = fate_eval.inverse_standardize(tgt_cells, dm_scaler)
                w2_raw = fate_eval.compute_w2(ep_raw, tgt_raw, device=device)

            ep_std = ep.std(0).mean()
            ep_mean_abs = np.abs(ep.mean(0)).mean()

            row = {
                "solver": solver, "n_steps": n_steps,
                "w2_scaled": float(w2_scaled), "w2_raw": float(w2_raw),
                "ep_std_mean": float(ep_std), "ep_mean_abs": float(ep_mean_abs),
            }
            solver_results.append(row)
            print(f"  {solver:8s} steps={n_steps:5d}  w2={w2_scaled:.6f}  "
                  f"ep_std={ep_std:.6f}  ep_mean_abs={ep_mean_abs:.6f}")

    import pandas as pd
    df_solver = pd.DataFrame(solver_results)
    df_solver.to_csv(os.path.join(output_dir, "test1_solver_sweep.csv"), index=False)
    print(f"\nSaved: {output_dir}/test1_solver_sweep.csv")

    # Find the best solver/steps combo
    best = df_solver.loc[df_solver["w2_scaled"].idxmin()]
    print(f"\nBest: {best['solver']} steps={int(best['n_steps'])}  w2={best['w2_scaled']:.6f}")
    worst = df_solver.loc[df_solver["w2_scaled"].idxmax()]
    print(f"Worst: {worst['solver']} steps={int(worst['n_steps'])}  w2={worst['w2_scaled']:.6f}")
    print(f"Ratio worst/best: {worst['w2_scaled']/best['w2_scaled']:.2f}x")

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 2: Identity reconstruction (zero-travel)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 2: IDENTITY RECONSTRUCTION (integrate t→t, expect zero error)")
    print("=" * 70)

    x_test = x0_t[:200]
    identity_results = []
    for solver in ["euler", "midpoint", "rk4", "dopri5"]:
        for n_steps in [10, 100, 1000]:
            node = NeuralODE(torch_wrapper(drift), solver=solver)
            t_mid = (t_start_w2 + t_end_w2) / 2.0
            with torch.no_grad():
                traj = node.trajectory(
                    x_test,
                    t_span=torch.linspace(t_mid, t_mid + 1e-6, n_steps, device=device),
                )
            err = (traj[-1] - x_test).abs()
            mean_err = err.mean().item()
            max_err = err.max().item()
            rel_err = (err / (x_test.abs() + 1e-10)).mean().item()

            row = {"solver": solver, "n_steps": n_steps,
                   "mean_abs_err": mean_err, "max_abs_err": max_err,
                   "mean_rel_err": rel_err}
            identity_results.append(row)
            print(f"  {solver:8s} steps={n_steps:5d}  mean_abs={mean_err:.2e}  "
                  f"max_abs={max_err:.2e}  mean_rel={rel_err:.2e}")

    df_identity = pd.DataFrame(identity_results)
    df_identity.to_csv(os.path.join(output_dir, "test2_identity.csv"), index=False)
    print(f"\nSaved: {output_dir}/test2_identity.csv")

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 3: Per-dimension W2 decomposition
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 3: PER-DIMENSION W2 DECOMPOSITION")
    print("=" * 70)

    # Use the best solver from Test 1
    best_solver = str(best["solver"])
    best_steps = int(best["n_steps"])
    print(f"Using best solver: {best_solver} steps={best_steps}")

    node = NeuralODE(torch_wrapper(drift), solver=best_solver)
    with torch.no_grad():
        traj = node.trajectory(
            x0_t,
            t_span=torch.linspace(t_start_w2, t_end_w2, best_steps, device=device),
        )
    ep_best = traj[-1].cpu().numpy()

    # Also get euler-100 endpoints for comparison
    node_euler = NeuralODE(torch_wrapper(drift), solver="euler")
    with torch.no_grad():
        traj_euler = node_euler.trajectory(
            x0_t,
            t_span=torch.linspace(t_start_w2, t_end_w2, 100, device=device),
        )
    ep_euler = traj_euler[-1].cpu().numpy()

    import ot as pot

    per_dim_results = []
    for d in range(n_dims):
        # Best solver
        ep_d = ep_best[:, d:d+1].astype(np.float64)
        tgt_d = tgt_cells[:, d:d+1].astype(np.float64)
        n_s, n_t = len(ep_d), len(tgt_d)
        M = pot.dist(ep_d, tgt_d, metric="sqeuclidean")
        w2_d_best = float(np.sqrt(max(pot.emd2(np.ones(n_s)/n_s, np.ones(n_t)/n_t, M), 0)))

        # Euler-100
        ep_d_e = ep_euler[:, d:d+1].astype(np.float64)
        M_e = pot.dist(ep_d_e, tgt_d, metric="sqeuclidean")
        w2_d_euler = float(np.sqrt(max(pot.emd2(np.ones(n_s)/n_s, np.ones(n_t)/n_t, M_e), 0)))

        row = {
            "dim": d,
            f"w2_{best_solver}": w2_d_best,
            "w2_euler100": w2_d_euler,
            "ep_std_best": float(ep_best[:, d].std()),
            "ep_std_euler": float(ep_euler[:, d].std()),
            "tgt_std": float(tgt_cells[:, d].std()),
            "ep_mean_best": float(ep_best[:, d].mean()),
            "ep_mean_euler": float(ep_euler[:, d].mean()),
            "tgt_mean": float(tgt_cells[:, d].mean()),
        }
        per_dim_results.append(row)
        print(f"  dim {d:2d}: w2_best={w2_d_best:.6f}  w2_euler={w2_d_euler:.6f}  "
              f"ep_std={ep_best[:,d].std():.6f}  tgt_std={tgt_cells[:,d].std():.6f}  "
              f"ep_mean={ep_best[:,d].mean():.6f}  tgt_mean={tgt_cells[:,d].mean():.6f}")

    df_perdim = pd.DataFrame(per_dim_results)
    df_perdim.to_csv(os.path.join(output_dir, "test3_per_dim_w2.csv"), index=False)
    print(f"\nSaved: {output_dir}/test3_per_dim_w2.csv")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Full W2 with best solver
    w2_full_best = fate_eval.compute_w2(ep_best, tgt_cells, device=device)
    w2_full_euler = fate_eval.compute_w2(ep_euler, tgt_cells, device=device)
    print(f"Full W2 (euler-100):      {w2_full_euler:.6f}")
    print(f"Full W2 ({best_solver}-{best_steps}): {w2_full_best:.6f}")
    print(f"Solver improvement:       {w2_full_euler/w2_full_best:.2f}x")
    print(f"Cluster reference:        ~0.007")
    print(f"Gap to cluster (best):    {w2_full_best/0.007:.1f}x")

    # Check if solver alone closes the gap
    if w2_full_best < 0.015:
        print("\n>> Solver swap SUBSTANTIALLY closes the gap. Consider using dopri5 at eval.")
    elif w2_full_best < 0.025:
        print("\n>> Solver helps but gap remains. Retraining on DM_EigenVectors_scaled recommended.")
    else:
        print("\n>> Solver has minimal effect. Model quality is the bottleneck. Retrain needed.")

    # Per-dim summary
    total_w2_sq = sum(r[f"w2_{best_solver}"]**2 for r in per_dim_results)
    print(f"\nPer-dim W2 contribution (fraction of sum-of-squares):")
    for r in sorted(per_dim_results, key=lambda x: x[f"w2_{best_solver}"], reverse=True):
        frac = r[f"w2_{best_solver}"]**2 / total_w2_sq * 100
        print(f"  dim {r['dim']:2d}: {frac:5.1f}%  w2={r[f'w2_{best_solver}']:.6f}  "
              f"std_ratio(ep/tgt)={r['ep_std_best']/max(r['tgt_std'],1e-10):.3f}")

    print(f"\nAll results saved to: {output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
