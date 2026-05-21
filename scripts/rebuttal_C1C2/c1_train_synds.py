#!/usr/bin/env python
"""Train one pde_params model on the 5-D synthetic FP dataset using the
Syn_DS reader (matches Sup Fig 2 / 2.train_model.py exactly: asymmetric
channels and deterministic timepoint hold-out).

Usage:
    c1_train_synds.py --lambda_growth 3 --seed 0 \
        --save_dir /abs/path/to/lambdag_3/seed_0 [--max_epochs 300]

`save_dir` is the literal leaf directory; nothing gets prepended/appended.
Two ModelCheckpoint callbacks save top-2 by val_loss and top-2 by total_loss
to `save_dir/lightning_logs/checkpoints/` with prefixes `best_val-*` and
`best_tot-*`.
"""
import argparse
import json
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning import callbacks
from torch.utils.data import DataLoader

from pseudodynamics import models, reader


SEEN = [0, 1, 2, 4, 6, 8, 10]   # exact Sup Fig 2 train timepoints
VAL  = [5, 7]                    # held-out validation timepoints
TEST = [3, 9]                    # held-out test timepoints


class DictSyn_DS(reader.Syn_DS):
    """Adapter: Syn_DS returns a tuple, but pde_params.training_step indexes
    a dict. cfm_x0/cfm_x1 are gated by `cfm_weight > 0` (we keep =0)."""
    def __getitem__(self, i):
        s, t, tp1, ut, utp1, deltax = super().__getitem__(i)
        return dict(s=s, t=t, tp1=tp1, ut=ut, utp1=utp1, deltax=deltax)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lambda_growth", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--save_dir", type=Path, required=True)
    p.add_argument("--max_epochs", type=int, default=300)
    p.add_argument(
        "--data_dir",
        type=Path,
        default=Path(
            "/rds/user/wz369/hpc-work/pseudodynamics_plus/"
            "data/synthesized_data/5Dim_ncs_synparam_Jan23"
        ),
    )
    p.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    pl.seed_everything(args.seed, workers=True)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    init = torch.load(args.data_dir / "syn_init_condition_0-4.ckpt",
                      map_location="cpu", weights_only=False)
    res = torch.load(args.data_dir / "syn_result_0-4.ckpt",
                     map_location="cpu", weights_only=False)
    cellstate = init["cellstate"]
    u = res["u_simulate"]
    t_arr = res["integrate_time"]
    dx = res["delta_x"]

    train_DS = DictSyn_DS(cellstate=cellstate, density=u[SEEN],
                          integrate_time=t_arr[SEEN], deltax=dx, batchsize=128)
    val_DS = DictSyn_DS(cellstate=cellstate, density=u[VAL],
                        integrate_time=t_arr[VAL], deltax=dx, batchsize=512)
    train_DL = DataLoader(train_DS, batch_size=None, num_workers=args.num_workers)
    val_DL = DataLoader(val_DS, batch_size=None, num_workers=args.num_workers)

    model = models.pde_params(
        channels=[6, 16, 16, 1],
        g_channels=[6, 16, 16, 1],
        v_channels=[6, 32, 16, 5],
        D_channels=[6, 32, 1],
        activation_fn="Tanh",
        ode_tol=1e-4,
        D_penalty=0.1,
        deltax_weight=1,
        weight_intensity=3,
        time_scale_factor=1.0,
        time_sensitive=True,
        growth_weight=args.lambda_growth,
        R_weight=None,
        cfm_weight=0,
        D_var_weight=0,
        neuralode_weight=None,
        lr=3e-4,
    )

    ckpt_dir = args.save_dir / "lightning_logs" / "checkpoints"
    cb_val = callbacks.ModelCheckpoint(
        dirpath=ckpt_dir, filename="best_val-{epoch}-{val_loss:.6f}",
        monitor="val_loss", mode="min", save_top_k=2,
    )
    cb_tot = callbacks.ModelCheckpoint(
        dirpath=ckpt_dir, filename="best_tot-{epoch}-{total_loss:.6f}",
        monitor="total_loss", mode="min", save_top_k=2,
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=[0] if args.accelerator == "gpu" else None,
        default_root_dir=str(args.save_dir),
        max_epochs=args.max_epochs,
        enable_progress_bar=False,
        callbacks=[cb_val, cb_tot],
    )

    (args.save_dir / "run_config.json").write_text(json.dumps(dict(
        lambda_growth=args.lambda_growth, seed=args.seed,
        seen=SEEN, val=VAL, test=TEST, max_epochs=args.max_epochs,
        channels=dict(u=[6, 16, 16, 1], g=[6, 16, 16, 1],
                      v=[6, 32, 16, 5], D=[6, 32, 1]),
        hp=dict(D_penalty=0.1, deltax_weight=1, weight_intensity=3,
                ode_tol=1e-4, time_sensitive=True, lr=3e-4),
    ), indent=2))

    trainer.fit(model, train_dataloaders=train_DL, val_dataloaders=val_DL)

    best_val = cb_val.best_model_score.item() if cb_val.best_model_score is not None else float("nan")
    best_tot = cb_tot.best_model_score.item() if cb_tot.best_model_score is not None else float("nan")
    print(f"DONE  best_val_loss={best_val:.6f}  best_total_loss={best_tot:.6f}")


if __name__ == "__main__":
    main()
