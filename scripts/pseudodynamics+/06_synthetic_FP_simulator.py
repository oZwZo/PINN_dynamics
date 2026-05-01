import os
import argparse
import pickle

import numpy as np
import pandas as pd
import anndata as ad


# ---------------------------------------------------------------------------
# Physical field definitions
# ---------------------------------------------------------------------------

def velocity(xy):
    """Drift v(x,y) = (-y, x) - 0.3*(x, y)   shape: (..., 2)"""
    x, y = xy[..., 0], xy[..., 1]
    vx = -y - 0.3 * x
    vy =  x - 0.3 * y
    return np.stack([vx, vy], axis=-1)


def diffusion(xy):
    """D(x,y) = 0.05 + 0.45 * exp(-(x^2+y^2)/0.09)   scalar per point"""
    r2 = xy[..., 0] ** 2 + xy[..., 1] ** 2
    return 0.05 + 0.45 * np.exp(-r2 / (0.3 ** 2))


def growth(xy):
    """g(x,y) = 0.5 * exp(-(x^2+y^2)/0.16)   scalar per point"""
    r2 = xy[..., 0] ** 2 + xy[..., 1] ** 2
    return 0.5 * np.exp(-r2 / (0.4 ** 2))


# ---------------------------------------------------------------------------
# Euler–Maruyama step
# ---------------------------------------------------------------------------

def em_step(pos, dt, rng):
    """Single EM step with reflection at [-1,1]^2."""
    v = velocity(pos)                       # (N, 2)
    D = diffusion(pos)[:, None]             # (N, 1)
    noise = rng.standard_normal(pos.shape)  # (N, 2)
    pos_new = pos + v * dt + np.sqrt(2.0 * D * dt) * noise
    # reflect at domain boundary [-1, 1]^2
    pos_new = np.clip(pos_new, -1.0, 1.0)
    return pos_new


# ---------------------------------------------------------------------------
# Population growth (birth-only)
# ---------------------------------------------------------------------------

def apply_births(pos, dt, rng):
    """Duplicate each particle with probability g(x)*dt; return updated array."""
    g = growth(pos)
    p_birth = np.clip(g * dt, 0.0, 1.0)
    born = rng.random(len(pos)) < p_birth
    if born.any():
        pos = np.concatenate([pos, pos[born]], axis=0)
    return pos


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def simulate(n_init, T, dt, snapshot_times, cap, seed):
    rng = np.random.default_rng(seed)

    # initial condition
    pos = rng.multivariate_normal(
        mean=[0.0, -0.7],
        cov=0.05 * np.eye(2),
        size=n_init,
    ).astype(np.float32)

    n_steps = int(round(T / dt))
    snap_set = set(snapshot_times)

    snapshots = {}        # t -> positions array
    true_counts = {}      # t -> true population (before cap/scaling)
    cumulative_scale = 1.0   # product of downsampling factors

    t_current = 0.0

    if t_current in snap_set:
        snapshots[t_current] = pos.copy()
        true_counts[t_current] = len(pos) * cumulative_scale

    for step in range(n_steps):
        pos = em_step(pos, dt, rng)
        pos = apply_births(pos, dt, rng)

        # cap
        if len(pos) > cap:
            factor = len(pos) / cap
            cumulative_scale *= factor
            idx = rng.choice(len(pos), size=cap, replace=False)
            pos = pos[idx]

        t_current = round((step + 1) * dt, 10)

        if t_current in snap_set:
            snapshots[t_current] = pos.copy()
            true_counts[t_current] = len(pos) * cumulative_scale

    return snapshots, true_counts


# ---------------------------------------------------------------------------
# Build AnnData
# ---------------------------------------------------------------------------

def build_adata(snapshots, true_counts, snapshot_times, seed):
    rng = np.random.default_rng(seed + 1)

    all_pos = []
    all_tp  = []

    for t in snapshot_times:
        p = snapshots[t]
        all_pos.append(p)
        all_tp.extend([float(t)] * len(p))

    X = np.concatenate(all_pos, axis=0).astype(np.float32)  # (N_total, 2)
    timepoints_arr = np.array(all_tp, dtype=np.float32)

    n_total = X.shape[0]

    # delta_x = drift evaluated at each saved particle position
    delta_x = velocity(X).astype(np.float32)

    # obs dataframe
    obs = pd.DataFrame(index=np.arange(n_total).astype(str))
    obs['timepoint_tx_days'] = timepoints_arr
    obs['timepoint'] = timepoints_arr

    # 70/15/15 train/val/test split
    idx = np.arange(n_total)
    rng.shuffle(idx)
    n_train = int(0.70 * n_total)
    n_val   = int(0.15 * n_total)
    split_labels = np.empty(n_total, dtype=object)
    split_labels[idx[:n_train]]             = 'train'
    split_labels[idx[n_train:n_train+n_val]] = 'val'
    split_labels[idx[n_train+n_val:]]        = 'test'
    obs['split'] = split_labels
    obs['Well']  = split_labels   # trainer also checks Well

    # pop dict — true population sizes (pre-cap, scaled)
    t_arr   = np.array(snapshot_times, dtype=float)
    var_arr = np.array([true_counts[t] for t in snapshot_times], dtype=float)
    mean_arr = var_arr.copy()
    std_arr  = np.sqrt(var_arr)
    n_lib_arr = np.ones(len(snapshot_times), dtype=int)

    pop_dict = {
        't':    t_arr,
        'mean': mean_arr,
        'var':  var_arr,
        'std':  std_arr,
        'n_lib': n_lib_arr,
    }

    ground_truth = {
        'D_field_function_name': 'centred_gaussian_0.05_to_0.5',
        'v_field_function_name': 'rotation_with_drift',
        'g_field_function_name': 'centred_gaussian_g_0.5',
    }

    adata = ad.AnnData(X=X, obs=obs)
    adata.obsm['cellstate'] = X.copy()
    adata.obsm['delta_x']   = delta_x
    adata.uns['pop']         = pop_dict
    adata.uns['ground_truth'] = ground_truth

    return adata


# ---------------------------------------------------------------------------
# Ground-truth grid pkl
# ---------------------------------------------------------------------------

def save_ground_truth_grid(out_dir):
    lin = np.linspace(-1.0, 1.0, 100)
    gx, gy = np.meshgrid(lin, lin)           # (100, 100)
    xy_grid = np.stack([gx, gy], axis=-1)    # (100, 100, 2)

    D_grid = diffusion(xy_grid)              # (100, 100)
    v_grid = velocity(xy_grid)               # (100, 100, 2)
    g_grid = growth(xy_grid)                 # (100, 100)

    payload = {
        'grid_x': gx,
        'grid_y': gy,
        'D_grid': D_grid,
        'v_grid': v_grid,
        'g_grid': g_grid,
    }

    pkl_path = os.path.join(out_dir, 'synthetic_FP_ground_truth.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(payload, f)
    return pkl_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Simulate 2D Fokker-Planck synthetic dataset'
    )
    parser.add_argument('--output', type=str, default='data/synthetic_FP.h5ad')
    parser.add_argument('--seed',   type=int, default=0)
    parser.add_argument('--n_init', type=int, default=5000)
    parser.add_argument('--T',      type=float, default=4.0)
    parser.add_argument('--dt',     type=float, default=0.01)
    args = parser.parse_args()

    snapshot_times = [0.0, 1.0, 2.0, 3.0, 4.0]
    cap = 50000

    print(f"Simulating with n_init={args.n_init}, T={args.T}, dt={args.dt}, seed={args.seed}")
    snapshots, true_counts = simulate(
        n_init=args.n_init,
        T=args.T,
        dt=args.dt,
        snapshot_times=snapshot_times,
        cap=cap,
        seed=args.seed,
    )

    print("True population counts per snapshot:")
    for t in snapshot_times:
        n_saved = len(snapshots[t])
        n_true  = true_counts[t]
        mean_pos = snapshots[t].mean(axis=0)
        print(f"  t={t:.1f}: saved={n_saved:6d}, true={n_true:.0f}, mean_pos=({mean_pos[0]:.4f}, {mean_pos[1]:.4f})")

    adata = build_adata(snapshots, true_counts, snapshot_times, seed=args.seed)

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    adata.write_h5ad(out_path)
    print(f"Saved AnnData to {out_path}")
    print(f"  shape: {adata.shape}")
    print(f"  obs keys: {list(adata.obs.columns)}")
    print(f"  obsm keys: {list(adata.obsm.keys())}")
    print(f"  uns['pop'] keys: {list(adata.uns['pop'].keys())}")

    out_dir = os.path.dirname(out_path)
    pkl_path = save_ground_truth_grid(out_dir)
    print(f"Saved ground-truth grid to {pkl_path}")


if __name__ == '__main__':
    main()
