#!/rds/user/wz369/hpc-work/LIBS/mamba/envs/mioflow/bin/python
"""
TIGON Training Script
======================
Trains a TIGON (neural ODE + unbalanced optimal transport) model on prepared
Klein dataset .npy files. Bypasses the interactive create_args() and drives
the TIGON training loop via argparse.

Usage
-----
python scripts/TIGON/02_train.py \
    --input_dir logs/TIGON/pca --dataset klein_train \
    --save_dir logs/TIGON/pca/model \
    --hidden_dim 64 --niters 5000 --gpu 0

python scripts/TIGON/02_train.py \
    --input_dir logs/TIGON/dm --dataset klein_train \
    --save_dir logs/TIGON/dm/model \
    --hidden_dim 64 --niters 5000 --gpu 0
"""

import os
import sys
import json
import time
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, '/rds/user/wz369/hpc-work/TIGON')

import random
from utility import UOT, initialize_weights, loaddata, train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


class Args:
    """Mimics the TIGON Args object built by create_args()."""
    pass


def parse_args():
    p = argparse.ArgumentParser(description="Train a TIGON model on Klein dataset")
    p.add_argument("--dataset", default="klein_train",
                   help=".npy filename without extension (default: klein_train)")
    p.add_argument("--timepoints", default="0.0,1.0,2.0",
                   help="Comma-separated normalized timepoints (default: 0.0,1.0,2.0)")
    p.add_argument("--niters", type=int, default=5000,
                   help="Number of training iterations (default: 5000)")
    p.add_argument("--lr", type=float, default=3e-3,
                   help="Learning rate (default: 3e-3)")
    p.add_argument("--num_samples", type=int, default=100,
                   help="Samples per training iteration (default: 100)")
    p.add_argument("--hidden_dim", type=int, default=64,
                   help="Hidden layer dimension (default: 64)")
    p.add_argument("--n_hiddens", type=int, default=4,
                   help="Number of hidden layers (default: 4)")
    p.add_argument("--activation", default="Tanh",
                   help="Activation function (default: Tanh)")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU device index (default: 0)")
    p.add_argument("--input_dir", required=True,
                   help="Directory containing .npy file")
    p.add_argument("--save_dir", required=True,
                   help="Output directory for checkpoints")
    p.add_argument("--seed", type=int, default=1,
                   help="Random seed (default: 1)")
    p.add_argument("--weight_decay", type=float, default=0.01,
                   help="Adam weight decay (default: 0.01)")
    p.add_argument("--top_k", type=int, default=2,
                   help="Keep top-K best checkpoints by loss (default: 2)")
    return p.parse_args()


def main():
    cli_args = parse_args()

    # ── build TIGON-compatible Args object ──
    args = Args()
    args.dataset = cli_args.dataset
    args.timepoints = [float(t.strip()) for t in cli_args.timepoints.split(",")]
    args.niters = cli_args.niters
    args.lr = cli_args.lr
    args.num_samples = cli_args.num_samples
    args.hidden_dim = cli_args.hidden_dim
    args.n_hiddens = cli_args.n_hiddens
    args.activation = cli_args.activation
    args.gpu = cli_args.gpu
    args.input_dir = cli_args.input_dir
    args.save_dir = cli_args.save_dir
    args.seed = cli_args.seed

    os.makedirs(args.save_dir, exist_ok=True)

    # ── setup logging to file ──
    fh = logging.FileHandler(os.path.join(args.save_dir, "train.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    log.addHandler(fh)

    log.info("TIGON Training")
    log.info(f"  dataset:     {args.dataset}")
    log.info(f"  timepoints:  {args.timepoints}")
    log.info(f"  niters:      {args.niters}")
    log.info(f"  lr:          {args.lr}")
    log.info(f"  num_samples: {args.num_samples}")
    log.info(f"  hidden_dim:  {args.hidden_dim}")
    log.info(f"  n_hiddens:   {args.n_hiddens}")
    log.info(f"  activation:  {args.activation}")
    log.info(f"  gpu:         {args.gpu}")
    log.info(f"  seed:        {args.seed}")
    log.info(f"  weight_decay:{cli_args.weight_decay}")
    log.info(f"  input_dir:   {args.input_dir}")
    log.info(f"  save_dir:    {args.save_dir}")

    # ── reproducibility ──
    torch.enable_grad()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')
    log.info(f"  device:      {device}")

    # ── load data ──
    data_train = loaddata(args, device)
    integral_time = args.timepoints
    log.info(f"  Loaded {len(data_train)} timepoints")
    for i, dt in enumerate(data_train):
        log.info(f"    tp {i}: {dt.shape}")

    time_pts = range(len(data_train))
    leave_1_out = []
    train_time = [x for i, x in enumerate(time_pts) if i != leave_1_out]

    # ── model ──
    in_out_dim = data_train[0].shape[1]
    func = UOT(in_out_dim=in_out_dim,
               hidden_dim=args.hidden_dim,
               n_hiddens=args.n_hiddens,
               activation=args.activation).to(device)
    func.apply(initialize_weights)
    log.info(f"  Model: UOT(in_out_dim={in_out_dim}, hidden_dim={args.hidden_dim}, "
             f"n_hiddens={args.n_hiddens}, activation={args.activation})")

    # ── ODE solver options ──
    options = {
        'method': 'Dopri5',
        'h': None,
        'rtol': 1e-3,
        'atol': 1e-5,
        'print_neval': False,
        'neval_max': 1000000,
        'safety': None,
    }

    # ── optimizer and scheduler ──
    optimizer = optim.Adam(func.parameters(), lr=args.lr,
                           weight_decay=cli_args.weight_decay)
    lr_adjust = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[args.niters - 400, args.niters - 200],
        gamma=0.5,
        last_epoch=-1
    )
    mse = nn.MSELoss()

    # ── tracking ──
    LOSS = []
    L2_1 = []
    L2_2 = []
    Trans = []
    Sigma = []

    # ── resume from checkpoint if exists ──
    # Prefer ckpt_final.pth (has optimizer state), fall back to ckpt.pth (symlink to best)
    resume_path = None
    for candidate in ['ckpt_final.pth', 'ckpt.pth']:
        p = os.path.join(args.save_dir, candidate)
        if os.path.exists(p):
            resume_path = p
            break
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device)
        func.load_state_dict(checkpoint['func_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        log.info(f"  Loaded checkpoint from {resume_path}")

    # ── save args as JSON ──
    args_dict = {
        "dataset": args.dataset,
        "timepoints": args.timepoints,
        "niters": args.niters,
        "lr": args.lr,
        "num_samples": args.num_samples,
        "hidden_dim": args.hidden_dim,
        "n_hiddens": args.n_hiddens,
        "activation": args.activation,
        "gpu": args.gpu,
        "seed": args.seed,
        "weight_decay": cli_args.weight_decay,
        "in_out_dim": in_out_dim,
        "top_k": cli_args.top_k,
    }
    with open(os.path.join(args.save_dir, "train_args.json"), "w") as f:
        json.dump(args_dict, f, indent=2)

    # ── top-K best checkpoint tracker ──
    # Each entry: (loss, iteration, filepath)
    top_k = cli_args.top_k
    best_ckpts = []  # sorted ascending by loss (best first)

    def maybe_save_topk(loss_val, itr):
        """Save checkpoint if loss is among the top-K best seen so far."""
        nonlocal best_ckpts

        # check if this loss qualifies
        if len(best_ckpts) >= top_k and loss_val >= best_ckpts[-1][0]:
            return  # not good enough

        # save new checkpoint
        ckpt_itr_path = os.path.join(args.save_dir, f'ckpt_best_itr{itr}.pth')
        torch.save({'func_state_dict': func.state_dict(),
                     'iteration': itr, 'loss': loss_val}, ckpt_itr_path)

        best_ckpts.append((loss_val, itr, ckpt_itr_path))
        best_ckpts.sort(key=lambda x: x[0])  # sort by loss ascending

        # evict worst if over capacity
        if len(best_ckpts) > top_k:
            _, evict_itr, evict_path = best_ckpts.pop()
            if os.path.exists(evict_path):
                os.remove(evict_path)
            log.info(f"  Top-{top_k} update: saved itr {itr} (loss={loss_val:.6f}), "
                     f"evicted itr {evict_itr}")
        else:
            log.info(f"  Top-{top_k} update: saved itr {itr} (loss={loss_val:.6f})")

    # ── training loop ──
    log.info("Starting training...")
    t_start = time.time()

    try:
        sigma_now = 1
        for itr in range(1, args.niters + 1):
            itr_start = time.time()
            optimizer.zero_grad()

            loss, loss1, sigma_now, L2_value1, L2_value2 = train_model(
                mse, func, args, data_train, train_time,
                integral_time, sigma_now, options, device, itr
            )

            loss.backward()
            optimizer.step()
            lr_adjust.step()

            loss_val = loss.item()
            LOSS.append(loss_val)
            Trans.append(loss1[-1].mean(0).item())
            Sigma.append(sigma_now)
            L2_1.append(L2_value1.tolist())
            L2_2.append(L2_value2.tolist())

            itr_time = time.time() - itr_start
            if itr % 10 == 0 or itr == 1:
                elapsed = time.time() - t_start
                log.info(f"Iter {itr:5d}/{args.niters}  loss={loss_val:.6f}  "
                         f"sigma={sigma_now:.4f}  iter_time={itr_time:.2f}s  "
                         f"elapsed={elapsed:.1f}s")

            maybe_save_topk(loss_val, itr)

    except KeyboardInterrupt:
        log.info("Training interrupted by user.")

    total_time = time.time() - t_start
    log.info(f"Training complete. Total time: {total_time:.1f}s")

    # ── log best checkpoints ──
    log.info(f"Top-{top_k} best checkpoints:")
    for rank, (bloss, bitr, bpath) in enumerate(best_ckpts, 1):
        log.info(f"  #{rank}  itr={bitr}  loss={bloss:.6f}  -> {bpath}")

    # ── symlink ckpt.pth -> best checkpoint for easy access ──
    if best_ckpts:
        best_path = best_ckpts[0][2]
        symlink_path = os.path.join(args.save_dir, 'ckpt.pth')
        if os.path.islink(symlink_path) or os.path.exists(symlink_path):
            os.remove(symlink_path)
        os.symlink(os.path.basename(best_path), symlink_path)
        log.info(f"Symlinked ckpt.pth -> {os.path.basename(best_path)}")

    # ── save final checkpoint (always, with full training history) ──
    final_path = os.path.join(args.save_dir, 'ckpt_final.pth')
    torch.save({
        'func_state_dict': func.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'LOSS': LOSS,
        'TRANS': Trans,
        'L2_1': L2_1,
        'L2_2': L2_2,
        'Sigma': Sigma,
    }, final_path)
    log.info(f"Saved final checkpoint -> {final_path}")


if __name__ == "__main__":
    main()
