#!/rds/user/wz369/hpc-work/LIBS/mamba/envs/mioflow/bin/python
"""
TrajectoryNet Training Wrapper
================================
Thin wrapper that calls `python -m TrajectoryNet.main` via subprocess
to avoid argparse conflicts. Saves train_args.json for eval reproducibility.

Usage
-----
python scripts/TrajectoryNet/02_train.py \
    --data_path logs/TrajectoryNet/pca30/klein_train.npz \
    --embedding_name X_pca --max_dim 30 \
    --save_dir logs/TrajectoryNet/pca30/model \
    --dims 128-128-128 --niters 10000 --gpu 0

python scripts/TrajectoryNet/02_train.py \
    --data_path logs/TrajectoryNet/dm10/klein_train.npz \
    --embedding_name DM_EigenVectors --max_dim 10 \
    --save_dir logs/TrajectoryNet/dm10/model \
    --dims 64-64-64 --niters 10000 --gpu 0
"""

import os
import sys
import json
import argparse
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train TrajectoryNet via subprocess (avoids parser conflicts)"
    )
    p.add_argument("--data_path", required=True,
                   help="Path to TrajectoryNet NPZ file")
    p.add_argument("--embedding_name", required=True,
                   help="Embedding key in NPZ (must match 01_prepare_data --obsm_key)")
    p.add_argument("--max_dim", type=int, required=True,
                   help="Number of dimensions (must match 01_prepare_data --n_dims)")
    p.add_argument("--save_dir", required=True,
                   help="Directory to save model checkpoints")
    p.add_argument("--niters", type=int, default=10000,
                   help="Training iterations (default: 10000)")
    p.add_argument("--dims", type=str, default="128-128-128",
                   help="Hidden layer dims, e.g. 128-128-128 (default: 128-128-128)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate (default: 1e-3)")
    p.add_argument("--batch_size", type=int, default=1000,
                   help="Batch size (default: 1000)")
    p.add_argument("--training_noise", type=float, default=0.1,
                   help="Training noise std (default: 0.1)")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU device index (default: 0)")
    p.add_argument("--save_freq", type=int, default=1000,
                   help="Checkpoint save frequency (default: 1000)")
    p.add_argument("--time_scale", type=float, default=0.5,
                   help="Time scale for integration (default: 0.5)")
    p.add_argument("--layer_type", type=str, default="concatsquash",
                   help="ODE layer type (default: concatsquash)")
    p.add_argument("--nonlinearity", type=str, default="tanh",
                   help="Activation function (default: tanh)")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Build TrajectoryNet command
    cmd = [
        sys.executable, "-m", "TrajectoryNet.main",
        "--dataset", os.path.abspath(args.data_path),
        "--embedding_name", args.embedding_name,
        "--max_dim", str(args.max_dim),
        "--dims", args.dims,
        "--niters", str(args.niters),
        "--lr", str(args.lr),
        "--batch_size", str(args.batch_size),
        "--training_noise", str(args.training_noise),
        "--gpu", str(args.gpu),
        "--save", os.path.abspath(args.save_dir),
        "--save_freq", str(args.save_freq),
        "--time_scale", str(args.time_scale),
        "--layer_type", args.layer_type,
        "--nonlinearity", args.nonlinearity,
    ]
    # NOTE: no --whiten (we pre-standardize in 01_prepare_data.py)
    # NOTE: no --use_growth (base CNF only)

    log.info("Running TrajectoryNet training:")
    log.info(f"  Command: {' '.join(cmd)}")

    # Save training args for reproducibility before training
    train_args = {
        "data_path": os.path.abspath(args.data_path),
        "embedding_name": args.embedding_name,
        "max_dim": args.max_dim,
        "dims": args.dims,
        "niters": args.niters,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "training_noise": args.training_noise,
        "gpu": args.gpu,
        "save_dir": os.path.abspath(args.save_dir),
        "save_freq": args.save_freq,
        "time_scale": args.time_scale,
        "layer_type": args.layer_type,
        "nonlinearity": args.nonlinearity,
        "whiten": False,
        "use_growth": False,
    }
    args_path = os.path.join(args.save_dir, "train_args.json")
    with open(args_path, "w") as f:
        json.dump(train_args, f, indent=2)
    log.info(f"  Saved train_args.json -> {args_path}")

    # Run training
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        log.error(f"TrajectoryNet training failed with return code {result.returncode}")
        sys.exit(result.returncode)

    # Verify checkpoint exists
    ckpt_path = os.path.join(args.save_dir, "checkpt.pth")
    if os.path.exists(ckpt_path):
        log.info(f"  Best checkpoint saved -> {ckpt_path}")
    else:
        log.warning(f"  No best checkpoint found at {ckpt_path}")
        # Check for periodic checkpoints
        import glob
        ckpts = glob.glob(os.path.join(args.save_dir, "checkpt-*.pth"))
        if ckpts:
            log.info(f"  Found periodic checkpoints: {ckpts}")
        else:
            log.error("  No checkpoints found at all!")

    log.info("Done.")


if __name__ == "__main__":
    main()
