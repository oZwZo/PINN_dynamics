#!/usr/bin/env python
"""
C2 — NN extrapolation diagnostic.

For each trained `pde_params` checkpoint, sample "negative" cell-states at
concentric shells of increasing distance from the observed support, evaluate
the surrogate density u_θ on them, and report a decay curve.

Strategy (per round-2 DeepSeek revisions):
  - For every observed cell s_i compute its mean k-NN distance d_i in the
    cellstate space (k=10). σ_ref := median(d_i).
  - For α ∈ {0.5, 1, 2, 4}, sample
        s_neg = s_i + α · σ_ref · ε,  ε ~ N(0, I)
    (250 ε draws per cell, then sub-sample 10 000 per shell).
  - Filter: keep s_neg with min-distance to S_obs ≥ 0.5 · σ_ref (cKDTree).
  - Report
        ratio(α) = quantile_95(|u_θ(s_neg^α)|) / max(quantile_95(|u_θ(s_obs)|), 1e-8)
    plus the *minimum* min-distance per shell (sanity check).
Pass criterion:
  ratio(α=2) < 0.1 AND ratio monotonically non-increasing for α in {1, 2, 4}.
  α=0.5 is reported but excluded from the monotonicity test
  (cells at that radius may have legitimate non-zero density).

Usage:
  c2_extrapolation_diagnostic.py
      --ckpt path/to/model.ckpt
      --h5ad data/synthetic_FP_5D.h5ad
      [--cellstate-key X_data]
      [--time-col time]
      [--out-dir logs/synthetic_FP_5D_ablation/eval/c2/<arm>_<seed>]
The script can also be pointed at the Klein, Ery_Mk, HSPC checkpoints by
overriding --cellstate-key and --time-col.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.spatial import cKDTree

import pseudodynamics as pdp                         # noqa: F401
from pseudodynamics import models


SHELLS = [0.5, 1.0, 2.0, 4.0]
N_TOTAL_PER_SHELL = 10_000
K_NN = 10
MIN_DIST_FACTOR = 0.5    # τ in plan: keep s_neg if min_dist ≥ τ · σ_ref


def compute_sigma_ref(s_obs: np.ndarray, k: int = K_NN) -> float:
    tree = cKDTree(s_obs)
    d, _ = tree.query(s_obs, k=k + 1)         # +1 because the first neighbour is self
    mean_d = d[:, 1:].mean(axis=1)
    return float(np.median(mean_d))


def sample_shell(s_obs: np.ndarray, alpha: float, sigma_ref: float, rng: np.random.Generator) -> np.ndarray:
    """Sample `N_TOTAL_PER_SHELL` perturbed points around random cells at L2 radius
    exactly `alpha * sigma_ref`. Direction is uniform on the unit sphere (Gaussian
    then normalise), so α directly controls the shell radius regardless of dim."""
    n, d = s_obs.shape
    cell_idx = rng.integers(0, n, size=N_TOTAL_PER_SHELL)
    eps = rng.standard_normal(size=(N_TOTAL_PER_SHELL, d))
    eps /= np.linalg.norm(eps, axis=-1, keepdims=True) + 1e-12     # unit-sphere directions
    return s_obs[cell_idx] + alpha * sigma_ref * eps


def filter_by_min_dist(s_obs: np.ndarray, s_neg: np.ndarray, tau: float) -> np.ndarray:
    tree = cKDTree(s_obs)
    d_min, _ = tree.query(s_neg, k=1)
    return s_neg[d_min >= tau], d_min[d_min >= tau]


def normalise_time(t: np.ndarray, mode: str = "min_minus") -> np.ndarray:
    if mode == "min_minus":
        return t - t.min()
    if mode in (None, "False", "false", "none"):
        return t
    raise ValueError(f"unknown norm_time={mode}")


def evaluate_u_theta(model, s: np.ndarray, t_value: float) -> np.ndarray:
    """Run u_θ over arbitrary cellstates at a single (already-normalised) time."""
    model.eval()
    s_t = torch.tensor(s, dtype=torch.float32)
    t_t = torch.full((s.shape[0], 1), float(t_value), dtype=torch.float32)
    with torch.no_grad():
        out = model.u(s_t, t_t)
    return out.detach().cpu().numpy().reshape(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--cellstate-key", default="X_data")
    parser.add_argument("--time-col", default="time")
    parser.add_argument("--norm-time", default="min_minus")
    parser.add_argument("--time-value", type=float, default=None,
                        help="If given, evaluate u_θ at this single (normalised) time; else use the median observed time")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    adata = sc.read_h5ad(args.h5ad)
    s_obs = np.asarray(adata.obsm[args.cellstate_key], dtype=np.float32)
    times = np.asarray(adata.obs[args.time_col], dtype=np.float64)
    times_norm = normalise_time(times, args.norm_time)
    t_eval = float(np.median(times_norm)) if args.time_value is None else args.time_value

    model = models.pde_params.load_from_checkpoint(args.ckpt, map_location="cpu")

    # Reference: u_θ on the observed support at t_eval
    u_obs = evaluate_u_theta(model, s_obs, t_eval)
    denom = max(float(np.quantile(np.abs(u_obs), 0.95)), 1e-8)

    sigma_ref = compute_sigma_ref(s_obs)
    tau = MIN_DIST_FACTOR * sigma_ref

    rows = []
    for alpha in SHELLS:
        s_neg_raw = sample_shell(s_obs, alpha, sigma_ref, rng)
        s_neg, d_min = filter_by_min_dist(s_obs, s_neg_raw, tau)
        if len(s_neg) == 0:
            print(f"[α={alpha}] all samples rejected by min-distance filter (τ={tau:.3f})")
            continue
        u_neg = evaluate_u_theta(model, s_neg, t_eval)
        ratio = float(np.quantile(np.abs(u_neg), 0.95)) / denom
        rows.append(dict(
            alpha=alpha,
            sigma_ref=sigma_ref,
            n_neg_kept=len(s_neg),
            n_neg_raw=len(s_neg_raw),
            d_min_min=float(d_min.min()),
            d_min_median=float(np.median(d_min)),
            u_neg_q50=float(np.quantile(np.abs(u_neg), 0.5)),
            u_neg_q95=float(np.quantile(np.abs(u_neg), 0.95)),
            u_obs_q95=denom,
            ratio_q95=ratio,
        ))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "c2_shell_ratios.csv", index=False)

    # Pass / fail — explicit handling of missing shells
    df_test = df[df["alpha"] >= 1].sort_values("alpha").reset_index(drop=True)
    have_full = set(df_test["alpha"].tolist()) >= {1.0, 2.0, 4.0}
    monotone = bool(np.all(np.diff(df_test["ratio_q95"]) <= 0)) if len(df_test) >= 2 else False
    alpha_2_row = df_test[df_test["alpha"] == 2.0]
    pass_threshold = bool(len(alpha_2_row) and alpha_2_row["ratio_q95"].iloc[0] < 0.1)
    verdict = "pass" if (have_full and monotone and pass_threshold) else "fail"
    (out_dir / "c2_verdict.json").write_text(
        '{"monotone_alpha_ge_1": %s, "ratio_at_alpha_2_lt_0.1": %s, "verdict": "%s"}\n'
        % (str(monotone).lower(), str(pass_threshold).lower(), verdict)
    )

    # decay curve
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(df["alpha"], df["ratio_q95"], marker="o")
    ax.axhline(0.1, color="grey", ls="--", lw=0.8, label="target 0.1")
    ax.set_xlabel(r"shell radius $\alpha$  (in units of $\sigma_{\mathrm{ref}}$)")
    ax.set_ylabel(r"$q_{95}(|u_\theta(\mathrm{neg})|) / q_{95}(|u_\theta(\mathrm{obs})|)$")
    ax.set_title(f"C2 decay  —  verdict: {verdict}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "c2_decay_curve.png", dpi=150)
    plt.close()

    print(df.to_string(index=False))
    print(f"verdict = {verdict}")
    print(f"wrote {out_dir}/c2_shell_ratios.csv, c2_verdict.json, c2_decay_curve.png")


if __name__ == "__main__":
    main()
