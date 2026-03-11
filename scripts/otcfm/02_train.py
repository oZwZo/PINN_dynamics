"""
OT-CFM Training Script
========================
Train an Optimal-Transport Conditional Flow Matching model on the Klein dataset.

Usage
-----
python scripts/otcfm/02_train.py \
    --data_path data/klein/klein_addpop.h5ad \
    --obsm_key X_pca --n_dims 30 --width 128 \
    --save_dir logs/otcfm/pca30/model --niters 20000 --gpu 0 \
    --standardize

python scripts/otcfm/02_train.py \
    --data_path data/klein/klein_addpop.h5ad \
    --obsm_key DM_EigenVectors --n_dims 10 --width 128 \
    --save_dir logs/otcfm/dm10/model --niters 10000 --gpu 0 \
    --standardize
"""

import os
import sys
import json
import argparse
import logging

import numpy as np
import torch
from tqdm import tqdm

from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchcfm.models import MLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train OT-CFM on Klein dataset")
    p.add_argument("--data_path", required=True, help="Path to .h5ad file")
    p.add_argument("--obsm_key", default="X_pca", help="obsm key for cell embeddings")
    p.add_argument("--n_dims", type=int, default=30, help="Number of embedding dimensions")
    p.add_argument("--tp_col", default="timepoint_tx_days", help="Timepoint column in obs")
    p.add_argument("--save_dir", required=True, help="Directory to save model checkpoint")
    p.add_argument("--sigma", type=float, default=0.1, help="OT-CFM sigma parameter")
    p.add_argument("--width", type=int, default=128, help="MLP hidden width")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--batch_size", type=int, default=256, help="Batch size per timepoint pair")
    p.add_argument("--niters", type=int, default=10000, help="Training iterations")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--gpu", type=int, default=0, help="GPU device index")
    p.add_argument("--standardize", action="store_true", default=True,
                   help="Standardize embeddings (default: True)")
    p.add_argument("--no_standardize", action="store_true",
                   help="Disable standardization")
    p.add_argument("--log_freq", type=int, default=100, help="Log loss every N iters")
    p.add_argument("--val_freq", type=int, default=500, help="Validate every N iters")
    p.add_argument("--patience", type=int, default=2000,
                   help="Early stopping patience (iters without val improvement)")
    p.add_argument("--n_val_batches", type=int, default=10,
                   help="Number of batches to average for val loss")
    p.add_argument("--val_frac", type=float, default=0.1,
                   help="Fraction of training cells held out for validation")
    return p.parse_args()


def get_batch(FM, X, batch_size, n_times, device):
    """Build training batch from all consecutive timepoint pairs."""
    ts, xts, uts = [], [], []
    for t_start in range(n_times - 1):
        x0 = torch.from_numpy(
            X[t_start][np.random.randint(X[t_start].shape[0], size=batch_size)]
        ).float().to(device)
        x1 = torch.from_numpy(
            X[t_start + 1][np.random.randint(X[t_start + 1].shape[0], size=batch_size)]
        ).float().to(device)
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        ts.append(t + t_start)
        xts.append(xt)
        uts.append(ut)
    return torch.cat(ts), torch.cat(xts), torch.cat(uts)


def main():
    args = parse_args()
    if args.no_standardize:
        args.standardize = False

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Load data ──
    import scanpy as sc
    adata = sc.read_h5ad(args.data_path)
    train_mask = adata.obs["Well"] != 2
    train_ad = adata[train_mask]

    timepoints = sorted(train_ad.obs[args.tp_col].unique())
    n_times = len(timepoints)
    log.info(f"Timepoints: {timepoints}, n_times={n_times}")

    # Extract embeddings per timepoint
    X_raw = [
        train_ad[train_ad.obs[args.tp_col] == tp].obsm[args.obsm_key][:, :args.n_dims].astype(np.float32)
        for tp in timepoints
    ]
    for i, tp in enumerate(timepoints):
        log.info(f"  tp={tp}: {X_raw[i].shape[0]} cells")

    # ── Standardize ──
    all_train = np.concatenate(X_raw, axis=0)
    train_mean = all_train.mean(axis=0)
    train_std = np.clip(all_train.std(axis=0), 1e-6, None)

    if args.standardize:
        X_all = [(x - train_mean) / train_std for x in X_raw]
        log.info("Standardized embeddings")
    else:
        X_all = X_raw
        log.info("No standardization")

    np.savez(os.path.join(args.save_dir, "scaler.npz"),
             mean=train_mean, std=train_std)

    # ── Split training cells into train / val per timepoint ──
    rng = np.random.RandomState(args.seed)
    X = []
    X_val = []
    for i in range(n_times):
        n = X_all[i].shape[0]
        n_val = max(1, int(n * args.val_frac))
        perm = rng.permutation(n)
        X_val.append(X_all[i][perm[:n_val]])
        X.append(X_all[i][perm[n_val:]])
        log.info(f"  tp={timepoints[i]}: train={X[i].shape[0]}, val={X_val[i].shape[0]}")

    # ── Save train metadata for evaluation ──
    # NOTE: train_meta stores ALL training cells (train+val) for KNN at eval time
    train_embs = np.empty(n_times, dtype=object)
    train_cts = np.empty(n_times, dtype=object)
    for i, tp in enumerate(timepoints):
        tp_mask = train_ad.obs[args.tp_col] == tp
        train_embs[i] = X_all[i]
        train_cts[i] = train_ad[tp_mask].obs["Annotation"].values.astype(str)
    np.savez(os.path.join(args.save_dir, "train_meta.npz"),
             embeddings=train_embs, celltypes=train_cts)

    # Save test cells (standardized)
    test_ad = adata[~train_mask]
    test_emb_raw = test_ad.obsm[args.obsm_key][:, :args.n_dims].astype(np.float32)
    if args.standardize:
        test_emb = (test_emb_raw - train_mean) / train_std
    else:
        test_emb = test_emb_raw

    # Normalize timepoints to 0,1,...,n_times-1
    tp_map = {tp: i for i, tp in enumerate(timepoints)}
    test_tps = np.array([tp_map[tp] for tp in test_ad.obs[args.tp_col].values], dtype=np.float32)

    np.savez(os.path.join(args.save_dir, "test_cells.npz"),
             embeddings=test_emb,
             timepoints=test_tps,
             celltypes=test_ad.obs["Annotation"].values.astype(str),
             barcodes=test_ad.obs.index.values.astype(str))

    # Save config
    config = {
        "obsm_key": args.obsm_key,
        "n_dims": args.n_dims,
        "in_out_dim": args.n_dims,
        "timepoints": [float(t) for t in timepoints],
        "normalized_timepoints": list(range(n_times)),
        "standardize": args.standardize,
        "sigma": args.sigma,
        "width": args.width,
    }
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ── Build model ──
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=args.sigma)
    model = MLP(dim=args.n_dims, time_varying=True, w=args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    log.info(f"Model: MLP(dim={args.n_dims}, w={args.width})")
    log.info(f"FM: ExactOptimalTransportConditionalFlowMatcher(sigma={args.sigma})")

    # ── Validation helper ──
    @torch.no_grad()
    def compute_val_loss():
        model.eval()
        losses = []
        for _ in range(args.n_val_batches):
            t, xt, ut = get_batch(FM, X_val, args.batch_size, n_times, device)
            vt = model(torch.cat([xt, t[:, None]], dim=-1))
            losses.append(torch.mean((vt - ut) ** 2).item())
        model.train()
        return np.mean(losses)

    # ── Training loop ──
    loss_history = []
    val_loss_history = []
    best_val_loss = float("inf")
    best_iter = 0
    ckpt_path = os.path.join(args.save_dir, "ckpt.pt")

    model.train()
    for i in tqdm(range(args.niters), desc="OT-CFM"):
        optimizer.zero_grad()
        t, xt, ut = get_batch(FM, X, args.batch_size, n_times, device)
        vt = model(torch.cat([xt, t[:, None]], dim=-1))
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)
        if (i + 1) % args.log_freq == 0:
            log.info(f"  iter {i+1}/{args.niters}  loss={loss_val:.4f}")

        # ── Validation + checkpointing + early stopping ──
        if (i + 1) % args.val_freq == 0:
            vl = compute_val_loss()
            val_loss_history.append({"iter": i + 1, "val_loss": vl})
            log.info(f"  [val] iter {i+1}  val_loss={vl:.4f}  best={best_val_loss:.4f}")

            if vl < best_val_loss:
                best_val_loss = vl
                best_iter = i + 1
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                    "loss_history": loss_history,
                    "val_loss_history": val_loss_history,
                    "best_val_loss": best_val_loss,
                    "best_iter": best_iter,
                }, ckpt_path)
                log.info(f"  [val] New best! Saved checkpoint (iter {best_iter})")

            if (i + 1) - best_iter >= args.patience:
                log.info(f"  Early stopping at iter {i+1} "
                         f"(no improvement for {args.patience} iters, best={best_iter})")
                break

    # If no val checkpoint was saved yet (val_freq > niters), save final
    if not os.path.exists(ckpt_path):
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "loss_history": loss_history,
            "val_loss_history": val_loss_history,
            "best_val_loss": best_val_loss,
            "best_iter": best_iter,
        }, ckpt_path)

    # Save args as JSON too
    with open(os.path.join(args.save_dir, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    log.info(f"Final train loss: {loss_history[-1]:.4f}")
    log.info(f"Best val loss: {best_val_loss:.4f} at iter {best_iter}")
    log.info("Done.")


if __name__ == "__main__":
    main()
