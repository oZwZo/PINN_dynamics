"""
MIOFlow Evaluation Script
===========================
Evaluates a trained MIOFlow model on held-out test cells (Well == 2):
  1. EMD (Earth Mover's Distance)
  2. MMD (Maximum Mean Discrepancy)
  3. W2 (Wasserstein-2) distance
  4. Fate bias accuracy via KNN

Usage
-----
python scripts/MIOFlow/02_evaluate.py \
    --data_path data/klein/klein_addpop.h5ad \
    --model_dir results/MIOFlow/klein_addpop_pca30 \
    --obsm_key X_pca --n_dims 30

python scripts/MIOFlow/02_evaluate.py \
    --data_path data/klein/klein_addpop.h5ad \
    --model_dir results/MIOFlow/klein_addpop_dm10 \
    --obsm_key DM_EigenVectors --n_dims 10
"""

import os
import sys
import argparse
import logging

# ── tqdm monkey-patch ─────────────────────────────────────────────────────
import tqdm as _tqdm
import tqdm.notebook
tqdm.notebook.tqdm = _tqdm.tqdm

import numpy as np
import pandas as pd
import torch
import scanpy as sc
import ot as pot
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr

from MIOFlow.utils import set_seeds
from MIOFlow.models import make_model
from MIOFlow.eval import generate_points
from MIOFlow.losses import MMD_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained MIOFlow model")
    p.add_argument("--data_path", required=True, help="Path to .h5ad file")
    p.add_argument("--model_dir", required=True,
                   help="Path to trained model dir containing model_checkpoints.pt")
    p.add_argument("--obsm_key", default="X_pca",
                   help="obsm key used during training (default: X_pca)")
    p.add_argument("--n_dims", type=int, default=30,
                   help="Embedding dimensions used during training (default: 30)")
    p.add_argument("--tp_col", default="timepoint_tx_days",
                   help="Timepoint column in adata.obs (default: timepoint_tx_days)")
    p.add_argument("--well_col", default="Well",
                   help="Well column for train/test split (default: Well)")
    p.add_argument("--celltype_col", default="label_man",
                   help="Cell-type annotation column (default: label_man)")
    p.add_argument("--n_sims", type=int, default=10,
                   help="Number of simulation replicates (default: 10)")
    p.add_argument("--n_points", type=int, default=1000,
                   help="Number of points per simulation (default: 1000)")
    p.add_argument("--seed", type=int, default=10, help="Random seed (default: 10)")
    p.add_argument("--output", default=None,
                   help="Path to save evaluation CSV (default: model_dir/eval_results.csv)")
    return p.parse_args()


# ── Wasserstein-2 helper ─────────────────────────────────────────────────
def compute_w2(x_sim: np.ndarray, x_true: np.ndarray) -> float:
    """W2 distance via torchcfm or fallback to sliced W2."""
    try:
        from torchcfm.optimal_transport import wasserstein
        return float(wasserstein(
            torch.tensor(x_sim).float(),
            torch.tensor(x_true).float(),
            power=2, method="exact"
        ))
    except (ImportError, Exception):
        log.warning("torchcfm W2 failed; falling back to sliced W2")
        from scipy.stats import wasserstein_distance
        return float(np.mean([
            wasserstein_distance(x_sim[:, d], x_true[:, d])
            for d in range(x_sim.shape[1])
        ]))


# ── EMD helper ────────────────────────────────────────────────────────────
def compute_emd(x_sim: np.ndarray, x_true: np.ndarray) -> float:
    """Earth Mover's Distance via POT."""
    n_sim, n_true = x_sim.shape[0], x_true.shape[0]
    a = pot.unif(n_sim)
    b = pot.unif(n_true)
    M = pot.dist(x_sim, x_true, metric="euclidean")
    return float(pot.emd2(a, b, M))


# ── MMD helper ────────────────────────────────────────────────────────────
def compute_mmd(x_sim: np.ndarray, x_true: np.ndarray) -> float:
    """MMD via MIOFlow's MMD_loss."""
    mmd_fn = MMD_loss()
    return float(mmd_fn.forward(
        torch.tensor(x_sim).float(),
        torch.tensor(x_true).float()
    ))


# ── Fate bias helper ─────────────────────────────────────────────────────
def fate_bias_accuracy(
    x_sim: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test_final: np.ndarray,
    y_test_final: np.ndarray,
    k: int = 15,
) -> dict:
    """
    Assign simulated cells to cell types via KNN on train coords.
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
    output_path = args.output or os.path.join(args.model_dir, "eval_results.csv")

    # ── load adata ────────────────────────────────────────────────────────
    log.info(f"Loading adata from {args.data_path}")
    adata = sc.read_h5ad(args.data_path)

    train_mask = adata.obs[args.well_col].astype(int) != 2
    test_mask = adata.obs[args.well_col].astype(int) == 2
    adata_train = adata[train_mask]
    adata_test = adata[test_mask]
    log.info(f"  Train: {train_mask.sum()}  Test: {test_mask.sum()}")

    tp_values = sorted(adata.obs[args.tp_col].unique())
    log.info(f"  Timepoints: {tp_values}")

    # ── load model ────────────────────────────────────────────────────────
    ckpt_path = os.path.join(args.model_dir, "model_checkpoints.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    opts = ckpt["options"]

    model = make_model(
        opts["model_features"], opts["layers"],
        activation=opts["activation"], scales=None, use_cuda=use_cuda
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info(f"Model loaded: features={opts['model_features']}  layers={opts['layers']}")

    # ── build train DataFrame (for generate_points sampling) ──────────────
    obsm_key = args.obsm_key
    n_dims = args.n_dims

    emb_train = np.array(adata_train.obsm[obsm_key][:, :n_dims])
    tp_train = np.array(adata_train.obs[args.tp_col].values)
    df_train = pd.DataFrame(emb_train, columns=[f"d{i}" for i in range(1, n_dims + 1)])
    df_train["samples"] = tp_train.astype(np.int32)

    groups = sorted(df_train["samples"].unique())

    # ── prepare test data per timepoint ───────────────────────────────────
    test_data = {}
    for tp in tp_values:
        tp_mask = (adata_test.obs[args.tp_col] == tp).values
        n_cells = tp_mask.sum()
        if n_cells > 0:
            test_data[tp] = np.array(adata_test[tp_mask].obsm[obsm_key][:, :n_dims])
            log.info(f"  Test cells at tp={tp}: {n_cells}")
        else:
            log.info(f"  Test cells at tp={tp}: 0 (skipping)")

    if not test_data:
        raise ValueError("No test cells found at any timepoint!")

    # ── prepare train data for KNN fate bias ──────────────────────────────
    tp_last = tp_values[-1]
    train_last_mask = (adata_train.obs[args.tp_col] == tp_last).values
    x_train_last = np.array(adata_train[train_last_mask].obsm[obsm_key][:, :n_dims]).astype(np.float32)
    y_train_last = adata_train[train_last_mask].obs[args.celltype_col].values.astype(str)

    # Test cell types at final timepoint (for fate bias)
    test_last_mask = (adata_test.obs[args.tp_col] == tp_last).values
    x_test_last = np.array(adata_test[test_last_mask].obsm[obsm_key][:, :n_dims]).astype(np.float32)
    y_test_last = adata_test[test_last_mask].obs[args.celltype_col].values.astype(str)

    # ── run evaluation ────────────────────────────────────────────────────
    log.info(f"\nRunning {args.n_sims} simulation replicates ...")
    sample_time = [int(g) for g in groups]  # [2, 4, 6]

    all_results = []

    for rep in range(args.n_sims):
        set_seeds(args.seed + rep)

        # Generate points: samples n_points from earliest tp, integrates ODE
        generated = generate_points(
            model, df_train, n_points=args.n_points,
            use_cuda=use_cuda, samples_key="samples",
            sample_time=sample_time,
            autoencoder=None, recon=False
        )
        # generated shape: (len(sample_time), n_points, n_dims)

        for tp_idx, tp in enumerate(tp_values):
            if tp not in test_data:
                continue

            x_sim = generated[tp_idx].astype(np.float64)
            x_real = test_data[tp].astype(np.float64)

            # Subsample to manageable size for EMD
            n_eval = min(args.n_points, x_real.shape[0])
            rng = np.random.RandomState(args.seed + rep)
            real_idx = rng.choice(x_real.shape[0], size=n_eval, replace=False)
            x_real_sub = x_real[real_idx]
            x_sim_sub = x_sim[:n_eval]

            emd = compute_emd(x_sim_sub, x_real_sub)
            mmd = compute_mmd(x_sim_sub, x_real_sub)
            w2 = compute_w2(x_sim_sub, x_real_sub)

            # Fate bias only at final timepoint
            pearson_r = np.nan
            if tp == tp_last and x_test_last.shape[0] > 0:
                fb = fate_bias_accuracy(
                    x_sim=x_sim.astype(np.float32),
                    x_train=x_train_last,
                    y_train=y_train_last,
                    x_test_final=x_test_last,
                    y_test_final=y_test_last,
                )
                pearson_r = fb["pearson_r"]

            row = {
                "replicate": rep + 1,
                "timepoint": tp,
                "emd": emd,
                "mmd": mmd,
                "w2": w2,
                "pearson_r": pearson_r,
            }
            all_results.append(row)
            log.info(
                f"  rep {rep+1:2d}/{args.n_sims}  tp={tp}  "
                f"EMD={emd:.4f}  MMD={mmd:.4f}  W2={w2:.4f}  "
                f"r={pearson_r:.4f}" if not np.isnan(pearson_r) else
                f"  rep {rep+1:2d}/{args.n_sims}  tp={tp}  "
                f"EMD={emd:.4f}  MMD={mmd:.4f}  W2={w2:.4f}"
            )

    # ── aggregate and save ────────────────────────────────────────────────
    df_results = pd.DataFrame(all_results)

    # Summary statistics
    log.info(f"\n{'='*60}")
    for tp in sorted(df_results["timepoint"].unique()):
        tp_df = df_results[df_results["timepoint"] == tp]
        log.info(f"Timepoint {tp}:")
        log.info(f"  EMD: {tp_df['emd'].mean():.4f} +/- {tp_df['emd'].std():.4f}")
        log.info(f"  MMD: {tp_df['mmd'].mean():.4f} +/- {tp_df['mmd'].std():.4f}")
        log.info(f"  W2:  {tp_df['w2'].mean():.4f} +/- {tp_df['w2'].std():.4f}")
        if not tp_df["pearson_r"].isna().all():
            log.info(f"  Fate r: {tp_df['pearson_r'].mean():.4f} +/- {tp_df['pearson_r'].std():.4f}")
    log.info(f"{'='*60}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_results.to_csv(output_path, index=False)
    log.info(f"\nResults saved -> {output_path}")

    # Save last replicate's fate fractions
    if tp_last in test_data:
        set_seeds(args.seed + args.n_sims - 1)
        generated_last = generate_points(
            model, df_train, n_points=args.n_points,
            use_cuda=use_cuda, samples_key="samples",
            sample_time=sample_time,
            autoencoder=None, recon=False
        )
        tp_last_idx = tp_values.index(tp_last)
        fb_final = fate_bias_accuracy(
            x_sim=generated_last[tp_last_idx].astype(np.float32),
            x_train=x_train_last,
            y_train=y_train_last,
            x_test_final=x_test_last,
            y_test_final=y_test_last,
        )
        fate_rows = []
        for ct in fb_final["cell_types"]:
            fate_rows.append({
                "cell_type": ct,
                "simulated_fraction": fb_final["simulated_fractions"][ct],
                "observed_fraction": fb_final["observed_fractions"][ct],
            })
        fate_csv = output_path.replace(".csv", "_fate_fractions.csv")
        pd.DataFrame(fate_rows).to_csv(fate_csv, index=False)
        log.info(f"Fate fractions -> {fate_csv}")

    # Save generated trajectories
    try:
        traj_path = os.path.join(args.model_dir, "eval_generated.npy")
        np.save(traj_path, generated)
        log.info(f"Generated points -> {traj_path}")
    except Exception:
        pass

    log.info("Done.")


if __name__ == "__main__":
    main()
