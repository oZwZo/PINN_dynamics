"""
SF2M DM10 Diagnostic — Tests 2-3 only (Test 1 already completed)
"""
import os, sys, json, numpy as np, torch
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_dir = "logs/sf2m/dm10/model_r1"
    output_dir = os.path.join(model_dir, "diagnostics")
    os.makedirs(output_dir, exist_ok=True)

    from torchcfm.models import MLP
    from torchcfm.utils import torch_wrapper
    from torchdyn.core import NeuralODE

    config = json.load(open(os.path.join(model_dir, "config.json")))
    n_dims = config["n_dims"]
    drift = MLP(dim=n_dims, time_varying=True, w=config["width"]).to(device)
    ckpt = torch.load(os.path.join(model_dir, "ckpt.pt"), map_location=device, weights_only=False)
    drift.load_state_dict(ckpt["drift_model_state_dict"])
    drift.eval()

    test_data = np.load(os.path.join(model_dir, "test_cells.npz"), allow_pickle=True)
    test_emb = test_data["embeddings"]
    test_tps = test_data["timepoints"]
    norm_tps = config["normalized_timepoints"]

    tp1 = float(norm_tps[1])
    tp_last = float(norm_tps[-1])
    src_mask = np.isclose(test_tps, tp1, atol=0.1)
    tgt_mask = np.isclose(test_tps, tp_last, atol=0.1)

    rng = np.random.RandomState(42)
    mc = 2000
    src_full = test_emb[src_mask].astype(np.float32)
    tgt_full = test_emb[tgt_mask].astype(np.float32)
    src_idx = rng.choice(len(src_full), mc, replace=False)
    tgt_idx = rng.choice(len(tgt_full), mc, replace=False)
    src_cells = src_full[src_idx]
    tgt_cells = tgt_full[tgt_idx]

    x0_t = torch.tensor(src_cells, dtype=torch.float32).to(device)

    # ═══════════════════════════════════════════════════════════
    # TEST 2: Velocity field magnitude check (replaces identity test)
    # ═══════════════════════════════════════════════════════════
    print("=" * 70)
    print("TEST 2: VELOCITY FIELD MAGNITUDE vs DATA SCALE")
    print("=" * 70)

    import pandas as pd

    for t_val in [1.0, 1.25, 1.5, 1.75, 2.0]:
        t_in = torch.full((x0_t.shape[0], 1), t_val, device=device)
        with torch.no_grad():
            v = drift(torch.cat([x0_t, t_in], dim=-1))
        v_np = v.cpu().numpy()
        v_norm = np.linalg.norm(v_np, axis=1)
        x_norm = np.linalg.norm(src_cells, axis=1)
        print(f"  t={t_val:.2f}  |v| mean={v_norm.mean():.6f}  |v|/|x|={v_norm.mean()/x_norm.mean():.3f}  "
              f"v_per_dim_std={v_np.std(0).mean():.6f}")

    print(f"\n  Data scale: src_std={src_cells.std(0).mean():.6f}  tgt_std={tgt_cells.std(0).mean():.6f}")
    print(f"  If |v|/|x| >> 1 over dt=1, endpoints overshoot. If << 1, endpoints barely move.")

    # ═══════════════════════════════════════════════════════════
    # TEST 3: Per-dimension W2 decomposition
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 3: PER-DIMENSION W2 DECOMPOSITION")
    print("=" * 70)

    node = NeuralODE(torch_wrapper(drift), solver="dopri5")
    with torch.no_grad():
        traj = node.trajectory(
            x0_t,
            t_span=torch.linspace(tp1, tp_last, 100, device=device),
        )
    ep = traj[-1].cpu().numpy()

    import ot as pot

    per_dim_results = []
    for d in range(n_dims):
        ep_d = ep[:, d:d+1].astype(np.float64)
        tgt_d = tgt_cells[:, d:d+1].astype(np.float64)
        n_s, n_t = len(ep_d), len(tgt_d)
        M = pot.dist(ep_d, tgt_d, metric="sqeuclidean")
        w2_d = float(np.sqrt(max(pot.emd2(np.ones(n_s)/n_s, np.ones(n_t)/n_t, M), 0)))

        row = {
            "dim": d,
            "w2": w2_d,
            "ep_std": float(ep[:, d].std()),
            "tgt_std": float(tgt_cells[:, d].std()),
            "ep_mean": float(ep[:, d].mean()),
            "tgt_mean": float(tgt_cells[:, d].mean()),
            "std_ratio": float(ep[:, d].std() / max(tgt_cells[:, d].std(), 1e-10)),
            "mean_diff": float(abs(ep[:, d].mean() - tgt_cells[:, d].mean())),
        }
        per_dim_results.append(row)
        print(f"  dim {d:2d}: w2={w2_d:.6f}  ep_std={ep[:,d].std():.6f}  tgt_std={tgt_cells[:,d].std():.6f}  "
              f"ratio={row['std_ratio']:.3f}  mean_diff={row['mean_diff']:.6f}")

    df = pd.DataFrame(per_dim_results)
    df.to_csv(os.path.join(output_dir, "test3_per_dim_w2.csv"), index=False)

    # Summary
    total_w2_sq = sum(r["w2"]**2 for r in per_dim_results)
    print(f"\nPer-dim W2 contribution (fraction of sum-of-squares):")
    for r in sorted(per_dim_results, key=lambda x: x["w2"], reverse=True)[:5]:
        frac = r["w2"]**2 / total_w2_sq * 100
        print(f"  dim {r['dim']:2d}: {frac:5.1f}%  w2={r['w2']:.6f}  std_ratio={r['std_ratio']:.3f}  mean_diff={r['mean_diff']:.6f}")

    # Key diagnostic: are endpoints under-dispersed or shifted?
    avg_std_ratio = np.mean([r["std_ratio"] for r in per_dim_results])
    avg_mean_diff = np.mean([r["mean_diff"] for r in per_dim_results])
    print(f"\n  Average std ratio (ep/tgt): {avg_std_ratio:.3f}")
    print(f"  Average mean diff: {avg_mean_diff:.6f}")
    if avg_std_ratio < 0.7:
        print("  >> Endpoints are UNDER-DISPERSED — model collapses toward mean")
    elif avg_std_ratio > 1.3:
        print("  >> Endpoints are OVER-DISPERSED — model overshoots")
    else:
        print("  >> Dispersion is reasonable; gap is from mean shift")

    print("\nDone.")


if __name__ == "__main__":
    main()
