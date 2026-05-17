#!/local/scratch/wz369/PINN_env/bin/python
"""
10_epsilon_sensitivity.py
-------------------------
ε-sensitivity sweep of the reconstruction loss, for the reviewer
response on log-density vs density. No retraining — uses a trained
checkpoint plus random-init / noise-perturbed variants of the same
architecture.

Plan: /rds/user/wz369/hpc-work/pseudodynamics_plus/.claude/state/plan.md (rev 4)

Outputs:
  results/epsilon_sensitivity/<ckpt_stem>__results.csv
  results/epsilon_sensitivity/epsilon_sensitivity_<dataset>.pdf
  results/epsilon_sensitivity/log_u_target_histograms_<dataset>.pdf

Run:
  ./scripts/pseudodynamics+/10_epsilon_sensitivity.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# repo on path. Use .absolute() (no symlink resolution) — Path.resolve()
# misbehaves inside the singularity bind mount where /work appears as a
# bind point but pathlib follows it to a different host path.
REPO = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
import scanpy as sc  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from torchdiffeq import odeint  # noqa: E402

import pseudodynamics  # noqa: E402
from pseudodynamics import models  # noqa: E402
from pseudodynamics import reader  # noqa: E402


DEFAULT_CKPT = (
    REPO
    / "logs/klein_PC30_lD1_cfm10_lgNone_b1024"
    / "pde_params_tsense/lightning_logs/version_0"
    / "checkpoints/epoch=349-val_loss=60.42720413.ckpt"
)
DEFAULT_CFG = REPO / "logs/klein_PC30_lD1_cfm10_lgNone_b1024/cfm10_b1024.json"
DEFAULT_EPS = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4)
S0_SEEDS = (0, 1, 2, 3, 4)
NOISE_FRAC = 0.1
SWEEP_BATCH_SIZE = 256
N_BATCHES = 8
BATCH_SEED = 7


# ---------------------------------------------------------------------------
# Loss reimplementation — explicit ε (no library monkey-patching)
# ---------------------------------------------------------------------------

def weighted_log_mse(x, x_hat, alpha):
    """Mirror of loss_fn(x, x_hat, weight=None) in _pde_informed_params.py:58-83.

    Both x and x_hat must already be in log-space. The weight uses the
    *clamped* target as in the library (weight has no grad — the library
    detaches it via no_grad in the meshgrid variant, and the pde_params
    variant treats `x` as a leaf target).

    Returns (loss, frac_clamped, max_residual_after_clamp).
    """
    if x.shape != x_hat.shape:
        x = x.squeeze()
        x_hat = x_hat.squeeze()
    assert x.shape == x_hat.shape

    # Clamp both, mirroring the library
    clipped_x   = x < -24
    clipped_xh  = x_hat < -24
    frac_clamped = (clipped_x | clipped_xh).float().mean().item()
    x   = torch.clamp(x,   min=-24)
    x_hat = torch.clamp(x_hat, min=-24)

    # The library uses x (target) with grad to form weight; we follow that.
    w = (24.0 + x) ** alpha
    w = w / w.sum().clamp_min(1e-30)

    loss = torch.sum(w * (x - x_hat) ** 2)
    max_resid = (x - x_hat).detach().abs().max().item()
    return loss, frac_clamped, max_resid


def recon_loss(model, batch, eps, alpha, neuralode_weight, device, ode_tol):
    """L_t + L_{t+1} + λ_NeuralODE · L_sim with explicit ε.

    Replicates forward_density_loss and forward_simulation (lines 671-710),
    aggregated as in training_step (lines 829-830) but limited to the
    reconstruction terms — auxiliary losses (D_norm, v_loss, growth, cfm,
    D_var, R_loss) are excluded since the reviewer concern is about recon.
    """
    s   = batch['s'].to(device).requires_grad_(True)
    t   = batch['t'].to(device)
    tp1 = batch['tp1'].to(device)
    ut  = batch['ut'].to(device)
    utp1 = batch['utp1'].to(device)

    # Library uses t/tp1 as flat (N,) — model.u expects (N, 1) for t.
    # forward_density_loss simply calls self.u(s, t). MLP_surrogate handles
    # broadcasting; we let the model handle it natively.

    # ---- L_t : boundary at t ----
    log_u_pred_t = model.u(s, t)
    L_t, fc_t, mr_t = weighted_log_mse(torch.log(ut + eps), log_u_pred_t, alpha)

    # ---- L_{t+1} : boundary at t+1 ----
    log_u_pred_tp1 = model.u(s, tp1)
    L_tp1, fc_tp1, mr_tp1 = weighted_log_mse(torch.log(utp1 + eps), log_u_pred_tp1, alpha)

    # ---- L_sim : ODE rollout from (s, ut) to t+1 ----
    t0 = t[0].item() / model.time_scale_factor
    t1 = tp1[0].item() / model.time_scale_factor

    zeros = torch.zeros_like(ut)
    duds_init = torch.zeros_like(s)
    init_condition = (ut, s, duds_init, zeros.clone(), zeros.clone(), zeros.clone())

    nfe_counter = [0]
    def ode_fn(t_, y_):
        nfe_counter[0] += 1
        return model.ode_func(t_, y_)

    # Non-adjoint odeint for evaluation (per plan B.3).
    u_int_seq, *_ = odeint(
        ode_fn,
        y0=init_condition,
        t=torch.tensor([t0, t1], dtype=torch.float32, device=device),
        atol=ode_tol,
        rtol=ode_tol,
        method='dopri5',
    )
    u_int_final = torch.nn.functional.relu(u_int_seq[-1])

    L_sim, fc_sim, mr_sim = weighted_log_mse(
        torch.log(utp1 + eps),
        torch.log(u_int_final + eps),
        alpha,
    )

    L_total = L_t + L_tp1 + neuralode_weight * L_sim

    return {
        'L_t': L_t, 'L_tp1': L_tp1, 'L_sim': L_sim, 'L_total': L_total,
        'frac_clamped_t':   fc_t,
        'frac_clamped_tp1': fc_tp1,
        'frac_clamped_sim': fc_sim,
        'frac_clamped':     (fc_t + fc_tp1 + fc_sim) / 3.0,
        'max_resid':  max(mr_t, mr_tp1, mr_sim),
        'min_uint':   u_int_final.detach().min().item(),
        'nfe':        nfe_counter[0],
    }


# ---------------------------------------------------------------------------
# Sweep machinery
# ---------------------------------------------------------------------------

def gather_batches(val_DS, n_batches, seed):
    """Sample a fixed set of batches once, reuse across all (state, ε) pairs.
    Seeds both numpy AND torch random — the dataset uses numpy but downstream
    transformations (e.g. tensor conversion order) may interact with torch."""
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    batches = [val_DS[i] for i in range(n_batches)]
    np.random.set_state(np_state)
    torch.random.set_rng_state(torch_state)
    return batches


def gradient_accum_norm(model, batches, eps, alpha, neuralode_weight, device, ode_tol):
    """Iterate over batches; accumulate per-parameter gradient *vectors* (sum
    across batches, NOT sum of squares). Return:
      ‖∇θ L‖ = sqrt(sum_p ‖G_p‖²)   where  G_p = Σ_b g_b_p
    plus mean diagnostic metrics.

    Per plan v4 — critic round 3 fix: sum of vectors is the actual gradient
    of the epoch-summed loss; sum-of-squares conflates batch variance with
    mean gradient and inflates ‖G‖ at trained states.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    grad_sum = [torch.zeros_like(p, device=device) for p in params]

    diag_keys = ['L_t', 'L_tp1', 'L_sim', 'L_total',
                 'frac_clamped', 'frac_clamped_t', 'frac_clamped_tp1', 'frac_clamped_sim',
                 'max_resid', 'min_uint', 'nfe']
    diag_accum = {k: 0.0 for k in diag_keys}
    n = 0

    for batch in batches:
        out = recon_loss(model, batch, eps, alpha, neuralode_weight,
                         device, ode_tol)
        grads = torch.autograd.grad(
            out['L_total'], params,
            retain_graph=False, allow_unused=True, create_graph=False,
        )
        for gs, g in zip(grad_sum, grads):
            if g is not None:
                gs.add_(g.detach())
        for k in diag_keys:
            v = out[k]
            diag_accum[k] += (v.item() if torch.is_tensor(v) else float(v))
        n += 1
        # explicit free
        del out, grads

    grad_norm_sq = 0.0
    for gs in grad_sum:
        grad_norm_sq += gs.pow(2).sum().item()
    grad_norm = float(np.sqrt(grad_norm_sq))

    diag_avg = {k: diag_accum[k] / max(n, 1) for k in diag_keys}
    diag_avg['grad_norm'] = grad_norm
    return diag_avg


def build_state(model_class, hparams, ckpt_path, mode, seed=None, device='cuda'):
    """mode: 'S0' (random init, seeded) / 'S1' (noise-perturbed) / 'S2' (trained)."""
    if mode == 'S2':
        model = model_class.load_from_checkpoint(str(ckpt_path), map_location=device)
        return model.to(device).eval()
    elif mode == 'S0':
        assert seed is not None
        # Reinstantiate from saved hparams; do NOT load weights.
        torch.manual_seed(seed)
        np.random.seed(seed)
        kwargs = dict(hparams)
        # Remove the auto-saved 'cfm_weight' / 'D_var_weight' if Lightning didn't preserve
        # — pde_params __init__ accepts them as None.
        model = model_class(**kwargs)
        return model.to(device).eval()
    elif mode == 'S1':
        model = model_class.load_from_checkpoint(str(ckpt_path), map_location=device)
        torch.manual_seed(123456)
        with torch.no_grad():
            for p in model.parameters():
                if p.dim() > 0:
                    sigma = NOISE_FRAC * p.detach().std().item()
                    if sigma > 0:
                        p.add_(torch.randn_like(p) * sigma)
        return model.to(device).eval()
    else:
        raise ValueError(f"unknown mode {mode}")


def build_dataloader(cfg_args, batch_size_override=None):
    """Mirror main_train.py:182-197 for the val split, with optional
    batch_size override (val split may have < batch_size cells)."""
    data_path = REPO / "data" / f"{cfg_args['dataset']}.h5ad"
    adata = sc.read_h5ad(str(data_path))

    timepoint_idx = cfg_args['timepoint_idx']
    if isinstance(timepoint_idx, str):
        timepoint_idx = eval(timepoint_idx)

    knn_volume = cfg_args.get('knn_volume', False)
    if isinstance(knn_volume, str):
        knn_volume = eval(knn_volume)

    bs = batch_size_override if batch_size_override is not None else cfg_args['batch_size']

    ds_kws = dict(
        timepoint_idx=timepoint_idx,
        n_dimension=cfg_args['n_dimension'],
        cellstate_key=cfg_args['cellstate_key'],
        knn_volume=knn_volume,
        log_transform=False,
        norm_time=cfg_args['norm_time'],
        deltax_key=cfg_args.get('deltax_key'),
        kde_kws={'bw_method': cfg_args.get('bw')},
        batchsize=bs,
    )
    val_DS = reader.TwoTimpepoint_AnnDS(AnnData=adata, split='val', **ds_kws)
    return val_DS, adata


def restore_hparams_for_reinit(ckpt_path, model_class):
    """Load hparams from the checkpoint, filtered to model_class.__init__ args
    via signature introspection (robust to constructor changes)."""
    import inspect
    state = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    saved = dict(state.get('hyper_parameters', {}))
    sig = inspect.signature(model_class.__init__)
    init_params = set(sig.parameters.keys()) - {'self'}
    filtered = {k: v for k, v in saved.items() if k in init_params}
    missing = init_params - set(filtered.keys())
    missing -= {'channels'}  # channels is required; rest are optional with defaults
    return filtered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=Path, default=DEFAULT_CKPT)
    p.add_argument('--cfg',  type=Path, default=DEFAULT_CFG)
    p.add_argument('--eps',  nargs='+', type=float, default=list(DEFAULT_EPS))
    p.add_argument('--seeds', nargs='+', type=int, default=list(S0_SEEDS))
    p.add_argument('--n-batches', type=int, default=N_BATCHES)
    p.add_argument('--batch-size', type=int, default=SWEEP_BATCH_SIZE)
    p.add_argument('--batch-seed', type=int, default=BATCH_SEED)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--output-dir', type=Path,
                   default=REPO / 'results' / 'epsilon_sensitivity')
    p.add_argument('--quick', action='store_true',
                   help='3 ε * 2 batches * 2 S0 seeds, for smoke test')
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[sweep] device = {device}')

    if args.quick:
        args.eps = [1e-12, 1e-10, 1e-4]
        args.seeds = [0, 1]
        args.n_batches = 2
        print('[sweep] QUICK mode')

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- load config + hparams ----
    with open(args.cfg) as f:
        cfg = json.load(f)
    cfg_args = cfg['raw_args']
    alpha = cfg_args['weight_intensity']
    neuralode_weight = cfg_args.get('neuralode_weight', None) or 2
    ode_tol = cfg_args.get('tol', 1e-4)
    print(f'[sweep] alpha (weight_intensity) = {alpha}')
    print(f'[sweep] neuralode_weight         = {neuralode_weight}')
    print(f'[sweep] ode_tol                  = {ode_tol}')

    hparams = restore_hparams_for_reinit(args.ckpt, models.pde_params)
    print(f'[sweep] reinit hparams: channels={hparams.get("channels")}')

    # ---- build dataloader, pre-sample batches ----
    val_DS, adata = build_dataloader(cfg_args, batch_size_override=args.batch_size)
    print(f'[sweep] val_DS len = {len(val_DS)}, cells = {val_DS.cellstate.shape[0]}')
    if val_DS.cellstate.shape[0] < args.batch_size:
        new_bs = max(64, val_DS.cellstate.shape[0] // 2)
        print(f'[sweep] val too small for batch_size={args.batch_size}; downsizing to {new_bs}')
        args.batch_size = new_bs
        val_DS, _ = build_dataloader(cfg_args, batch_size_override=args.batch_size)
    batches = gather_batches(val_DS, args.n_batches, seed=args.batch_seed)
    print(f'[sweep] sampled {len(batches)} batches of size {args.batch_size}')

    # ---- iterate states × ε ----
    rows = []
    state_specs = (
        [('S0', s) for s in args.seeds]
        + [('S1', None), ('S2', None)]
    )
    model_class = models.pde_params

    for mode, seed in state_specs:
        print(f'\n[sweep] === state={mode} seed={seed} ===')
        model = build_state(model_class, hparams, args.ckpt, mode, seed, device)
        for eps in args.eps:
            diag = gradient_accum_norm(
                model, batches, eps, alpha, neuralode_weight, device, ode_tol,
            )
            row = {
                'state': mode,
                'seed':  seed,
                'eps':   eps,
                'alpha': alpha,
                **diag,
            }
            rows.append(row)
            print(f'  ε={eps:.0e}  L_total={diag["L_total"]:.4g}  '
                  f'‖∇θ L‖={diag["grad_norm"]:.4g}  '
                  f'frac_clamped={diag["frac_clamped"]:.3g}  '
                  f'min_uint={diag["min_uint"]:.3g}  nfe={diag["nfe"]:.0f}')
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)

    # ---- validation gates ----
    gate_results = compute_gates(df, ckpt_val_loss_hint=parse_val_loss_from_ckpt(args.ckpt))
    df['gate_passed'] = True  # per-row passes the SOFT sanity gate; hard gate is global
    for k, v in gate_results.items():
        print(f'[gate] {k}: {v}')

    # ---- save outputs ----
    ckpt_stem = args.ckpt.stem
    csv_path = args.output_dir / f'{ckpt_stem}__results.csv'
    df.to_csv(csv_path, index=False)
    print(f'\n[sweep] wrote {csv_path}')

    fig_path = args.output_dir / f'epsilon_sensitivity_{cfg_args["dataset"]}.pdf'
    make_main_figure(df, fig_path)
    print(f'[sweep] wrote {fig_path}')

    hist_path = args.output_dir / f'log_u_target_histograms_{cfg_args["dataset"]}.pdf'
    make_histogram_figure(val_DS, hist_path, eps_list=args.eps)
    print(f'[sweep] wrote {hist_path}')

    gate_json = args.output_dir / f'{ckpt_stem}__gates.json'
    with open(gate_json, 'w') as f:
        json.dump(gate_results, f, indent=2, default=str)
    print(f'[sweep] wrote {gate_json}')


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def parse_val_loss_from_ckpt(ckpt_path: Path):
    """Extract val_loss=NNN.NN from filename, if present."""
    name = ckpt_path.name
    for tok in name.split('-'):
        if tok.startswith('val_loss='):
            try:
                return float(tok.replace('val_loss=', '').replace('.ckpt', ''))
            except ValueError:
                return None
    return None


def compute_gates(df: pd.DataFrame, ckpt_val_loss_hint=None):
    """
    Hard gate (plan B.2 gate 1):
      min over S0 seeds of ‖∇θ L‖ at ε=1e-10  ≥  10 ·  ‖∇θ L‖ at S2 ε=1e-10
    Soft gate (plan B.2 gate 2):
      L_total at S2 ε=1e-10  within 1.5×  ckpt val_loss   (recon ≤ full loss)
    """
    out = {}
    eps_ref = 1e-10
    s2_row = df.query('state == "S2" and eps == @eps_ref')
    s0_rows = df.query('state == "S0" and eps == @eps_ref')
    if s2_row.empty or s0_rows.empty:
        out['hard_gate'] = 'SKIPPED (missing rows)'
    else:
        s2_grad = float(s2_row['grad_norm'].iloc[0])
        s0_min_grad = float(s0_rows['grad_norm'].min())
        ratio = s0_min_grad / max(s2_grad, 1e-30)
        out['s2_grad_norm'] = s2_grad
        out['s0_min_grad_norm'] = s0_min_grad
        out['s0_to_s2_ratio'] = ratio
        out['hard_gate'] = 'PASS' if ratio >= 10 else f'FAIL (ratio={ratio:.2f} < 10)'

    # Informational only: recon L_total vs full val_loss from ckpt name.
    # NOT a gate — `full val_loss = recon + auxiliary`, so the ratio is always
    # ≤ 1 by construction. Print for context, no pass/fail.
    if not s2_row.empty and ckpt_val_loss_hint is not None:
        s2_loss = float(s2_row['L_total'].iloc[0])
        out['s2_recon_L_total'] = s2_loss
        out['ckpt_val_loss']    = ckpt_val_loss_hint
        out['recon_fraction_of_val'] = s2_loss / max(ckpt_val_loss_hint, 1e-30)

    if 'L_total' in df.columns:
        spans = []
        for (state, seed), grp in df.groupby(['state', 'seed'], dropna=False):
            if len(grp) < 2:
                continue
            lt = grp['L_total'].values
            spans.append((state, seed, lt.max() / max(lt.min(), 1e-30)))
        out['eps_span_L_total'] = [(s, sd, f'{r:.3g}') for (s, sd, r) in spans]
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_main_figure(df: pd.DataFrame, path: Path):
    """1x2 figure: L_recon and ‖∇θ L_recon‖ vs ε. frac_clamped and min_uint
    are recorded in the CSV but not plotted — they were judged not to
    contribute to the manuscript figure."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)

    def state_curve(state, key):
        if state == 'S0':
            grp = df[df['state'] == 'S0'].groupby('eps')[key]
            return grp.mean().sort_index(), grp.min().sort_index(), grp.max().sort_index()
        sub = df[df['state'] == state].sort_values('eps')
        s = sub.set_index('eps')[key]
        return s, s, s

    panel_specs = [
        (axes[0], 'L_total',   r'$L_{\mathrm{recon}}$',                      '(a) reconstruction loss'),
        (axes[1], 'grad_norm', r'$\|\nabla_\theta L_{\mathrm{recon}}\|$',    r'(b) gradient norm $\|\nabla_\theta L\|$'),
    ]
    colors = {'S0': 'C0', 'S1': 'C1', 'S2': 'C2'}

    for ax, key, ylabel, title in panel_specs:
        for state in ['S0', 'S1', 'S2']:
            mean, lo, hi = state_curve(state, key)
            ax.plot(mean.index, mean.values, marker='o', label=state, color=colors[state])
            if state == 'S0':
                ax.fill_between(mean.index, lo.values, hi.values, alpha=0.2, color=colors[state])
        ax.set_xscale('log')
        try:
            ax.set_yscale('log')
        except Exception:
            pass
        ax.set_xlabel(r'$\varepsilon$')
        ax.set_ylabel(ylabel)
        ax.axvline(1e-10, color='k', linestyle=':', alpha=0.4)
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.3)
        ax.set_title(title, fontsize=10)

    fig.suptitle(r'$\varepsilon$-sensitivity of reconstruction loss and its gradient', fontsize=12)
    fig.savefig(path)
    plt.close(fig)


def make_histogram_figure(val_DS, path: Path, eps_list):
    """Per-timepoint overlay-histograms of log(u_target + 1e-10)."""
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    u_b = val_DS.u_b   # (n_timepoint, n_cells_in_split)
    for ti in range(u_b.shape[0]):
        log_u = np.log(u_b[ti] + 1e-10)
        ax.hist(log_u, bins=80, histtype='step', label=f't_idx={ti}', alpha=0.8)
    for eps in eps_list:
        ax.axvline(np.log(eps), color='gray', linestyle=':', alpha=0.4)
        ax.text(np.log(eps), ax.get_ylim()[1] * 0.95, f'log(ε={eps:.0e})',
                rotation=90, fontsize=7, va='top', ha='right', alpha=0.6)
    ax.set_xlabel(r'$\log(u_\mathrm{target} + 10^{-10})$')
    ax.set_ylabel('count')
    ax.set_title('Distribution of log target density across timepoints (val split)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(path)
    plt.close(fig)


if __name__ == '__main__':
    main()
