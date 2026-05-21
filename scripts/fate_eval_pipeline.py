"""
fate_eval_pipeline.py — Method-agnostic per-cell fate accuracy evaluation
==========================================================================

A general pipeline for evaluating how well any trajectory inference model
predicts the dominant terminal fate of individual starting cells, using
clone-traced ground-truth proportions (F_obs).

Design
------
The two core functions accept `model` and `simulation_func` as arguments,
making the pipeline method-agnostic.  `simulation_func` acts as a decorator
that wraps method-specific forward simulation behind a common interface:

    simulation_func(start_cells, model, n_sims, device) -> np.ndarray

         start_cells : ndarray (n_cells, n_dims)
         model       : any — passed through unchanged from the caller
         n_sims      : int  — trajectories per start cell (stochastic methods)
         device      : str  — torch device string, e.g. 'cuda:0'
         returns     : ndarray
                         (n_cells, n_sims, n_dims)  stochastic — one row of
                                                    n_sims endpoint positions
                                                    per start cell
                         (n_cells, n_dims)           deterministic — a single
                                                    endpoint per start cell;
                                                    n_sims is ignored

The same simulation_func factory pattern lets each 03_evaluate.py build its
own function once (capturing model-specific state) and pass it in:

    sim_fn = make_prescient_sim_fn()
    result = run_fate_evaluation(..., model=(net, config), simulation_func=sim_fn)

Public API
----------
  predict_fates_per_cell(start_cells, model, simulation_func,
                         knn, cell_types, n_sims, device)
      -> np.ndarray (n_cells, n_cell_types)

  compute_fate_accuracy(F_obs_df, F_hat_df)
      -> dict(accuracy, pearson_r, pearson_p, y_true, y_pred)

  run_fate_evaluation(start_cells, start_cell_ids, model, simulation_func,
                      F_obs, x_ref, y_ref,
                      cell_types=None, n_sims=100, k=20, device='cpu')
      -> dict(accuracy, pearson_r, pearson_p, n_start_cells,
              F_hat, F_obs_aligned, y_true, y_pred)

Provided simulation_func factories
------------------------------------
  make_prescient_sim_fn(num_steps=None)
      model = (net, config)           stochastic SDE (PRESCIENT)

  make_tigon_sim_fn(t_start, t_end)
      model = func                    ODE via TorchDiffEqPack (TIGON)

  make_trajectorynet_sim_fn(int_tp_start, int_tp_end)
      model = cnf_model               CNF (TrajectoryNet)

  make_otcfm_sim_fn(t_start, t_end)
      model = mlp_model               ODE via torchdyn (OT-CFM)

  make_sf2m_ode_sim_fn(t_start, t_end)
      model = drift_model             ODE via torchdyn (SF2M drift only)

  make_sf2m_sde_sim_fn(t_start, t_end, sigma, n_steps)
      model = (drift_model, score_model)   SDE via torchsde (SF2M)

Method-agnostic W2 helpers
---------------------------
  compute_w2(x_sim, x_true)
      Exact W2 distance via POT (ot.emd2)

  eval_w2_tasks(eval_tasks, simulate_fn, n_sims, n_sim_cells)
      Batch W2 evaluation over task list

Quick example
-------------
  from scripts.fate_eval_pipeline import run_fate_evaluation, make_prescient_sim_fn
  import pandas as pd

  F_obs = pd.read_csv("data/klein/F_obs.csv", index_col=0)
  F_obs.index = F_obs.index.astype(str)

  result = run_fate_evaluation(
      start_cells     = x_start,          # (n_cells, n_dims) float32
      start_cell_ids  = start_barcodes,   # list[str]
      model           = (net, config),
      simulation_func = make_prescient_sim_fn(),
      F_obs           = F_obs,
      x_ref           = x_train,          # (n_ref, n_dims) float32
      y_ref           = y_train,          # (n_ref,) str
      n_sims          = 100,
      k               = 20,
      device          = "cuda:0",
  )
  print(f"Accuracy : {result['accuracy']:.4f}")
  print(f"Pearson r: {result['pearson_r']:.4f}")
  # result['F_hat']         DataFrame (n_cells × n_cell_types)
  # result['F_obs_aligned'] DataFrame (n_cells × n_cell_types)
"""

import logging
import torch
import numpy as np
import scanpy as sc
import pandas as pd
import sklearn.metrics
from scipy.stats import pearsonr
from sklearn.neighbors import KNeighborsClassifier
log = logging.getLogger(__name__)


# ─── Core: predict fates ─────────────────────────────────────────────────────

def predict_fates_per_cell(
    start_cells: np.ndarray,
    model,
    simulation_func,
    knn: KNeighborsClassifier,
    cell_types: list,
    n_sims: int = 100,
    device: str = "cpu",
) -> np.ndarray:
    """
    Predict fate distribution for each start cell via simulation + KNN.

    Parameters
    ----------
    start_cells     : ndarray (n_cells, n_dims) — initial cell positions.
    model           : any model object, passed through to simulation_func.
    simulation_func : callable with signature
                        (start_cells, model, n_sims, device) -> ndarray
                      Returns (n_cells, n_sims, n_dims) for stochastic models,
                      or (n_cells, n_dims) for deterministic models.
    knn             : fitted KNeighborsClassifier mapping embedding → cell type.
    cell_types      : ordered list of cell-type strings (defines column order).
    n_sims          : trajectories per start cell (passed to simulation_func).
    device          : torch device string.

    Returns
    -------
    F_hat_arr : ndarray (n_cells, n_cell_types), rows sum to 1.
    """
    log.info(
        f"  simulation_func: {len(start_cells)} cells"
        f" × {n_sims} sims ..."
    )
    endpoints = np.asarray(simulation_func(start_cells, model, n_sims, device))
    print("endpoints shape :", endpoints.shape)
    n_cells  = len(start_cells)
    ct_index = {ct: i for i, ct in enumerate(cell_types)}

    # Deterministic: (n_cells, n_dims) → treat as a single sim per cell
    if endpoints.ndim == 2:
        endpoints = endpoints[:, np.newaxis, :]   # (n_cells, 1, n_dims)
    # Now guaranteed: (n_cells, n_sims_actual, n_dims)
    actual_sims = endpoints.shape[1]

    # Batch KNN: flatten → predict → reshape
    flat_pts   = endpoints.reshape(-1, endpoints.shape[-1])  # (n_cells*sims, n_dims)
    all_labels = knn.predict(flat_pts).reshape(n_cells, actual_sims)

    F_hat_arr = np.zeros((n_cells, len(cell_types)), dtype=np.float32)
    for i in range(n_cells):
        uniq, counts = np.unique(all_labels[i], return_counts=True)
        for ct, cnt in zip(uniq, counts):
            if ct in ct_index:
                F_hat_arr[i, ct_index[ct]] = cnt / actual_sims

    return F_hat_arr


# ─── Accuracy metric ─────────────────────────────────────────────────────────

def compute_fate_accuracy(
    F_obs_df: pd.DataFrame,
    F_hat_df: pd.DataFrame,
) -> dict:
    """
    Compare predicted vs. ground-truth per-cell fate distributions.

    Accuracy is the fraction of cells where the predicted dominant fate
    (argmax of F_hat row) matches the ground-truth dominant fate
    (argmax of F_obs row).  Cells with all-zero F_hat are assigned
    "Undifferentiated".

    Pearson r is computed over per-cell-type mean proportions
    (population-level distribution similarity).

    Parameters
    ----------
    F_obs_df : DataFrame (n_cells, n_cell_types) — ground-truth proportions.
    F_hat_df : DataFrame (n_cells, n_cell_types) — predicted proportions.
               Must share the same index as F_obs_df.

    Returns
    -------
    dict with keys: accuracy, pearson_r, pearson_p, y_true, y_pred.
    """
    shared_cols = sorted(set(F_obs_df.columns) & set(F_hat_df.columns))
    if not shared_cols:
        raise ValueError("F_obs and F_hat share no cell-type columns.")

    obs = F_obs_df[shared_cols]
    hat = F_hat_df[shared_cols]

    y_true = obs.idxmax(axis=1).tolist()
    y_pred = hat.idxmax(axis=1).tolist()

    # Assign "Undifferentiated" to cells where no fate was predicted
    zero_idx = np.where(hat.sum(axis=1).values == 0)[0]
    for idx in zero_idx:
        y_pred[idx] = "Undifferentiated"

    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)

    obs_mean = obs.mean(axis=0).values
    hat_mean = hat.mean(axis=0).values
    if obs_mean.std() < 1e-10 or hat_mean.std() < 1e-10:
        r, p = 0.0, 1.0
    else:
        r, p = pearsonr(obs_mean, hat_mean)

    return dict(accuracy=accuracy, pearson_r=r, pearson_p=p,
                y_true=y_true, y_pred=y_pred)


# ─── End-to-end pipeline ─────────────────────────────────────────────────────

def run_fate_evaluation(
    start_cells: np.ndarray,
    start_cell_ids,
    model,
    simulation_func,
    F_obs: pd.DataFrame,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    cell_types: list = None,
    n_sims: int = 100,
    k: int = 20,
    device: str = "cpu",
    knn: "KNeighborsClassifier | None" = None,
) -> dict:
    """
    End-to-end fate accuracy evaluation for any trajectory model.

    Ties together: (1) alignment of start cells to F_obs ground truth,
    (2) KNN fitting on reference data, (3) fate prediction via
    predict_fates_per_cell, (4) accuracy/Pearson-r via compute_fate_accuracy.

    Parameters
    ----------
    start_cells     : ndarray (n_cells, n_dims) — embedding of start cells.
    start_cell_ids  : array-like of str — barcodes for each start cell,
                      must align row-for-row with start_cells.
    model           : model object, passed through to simulation_func.
    simulation_func : callable — see predict_fates_per_cell docstring.
    F_obs           : DataFrame (cells × cell_types) — ground-truth fate
                      proportions indexed by cell barcode.
    x_ref           : ndarray (n_ref, n_dims) — reference embedding for KNN.
    y_ref           : ndarray (n_ref,) str — reference cell-type labels.
    cell_types      : list of cell-type names; defaults to sorted(F_obs.columns).
    n_sims          : trajectories per start cell.
    k               : KNN neighbours for cell-type assignment.
    device          : torch device string.

    Returns
    -------
    dict with keys:
        accuracy        float
        pearson_r       float
        pearson_p       float
        n_start_cells   int — cells actually evaluated (intersection with F_obs)
        F_hat           DataFrame (n_cells × n_cell_types)
        F_obs_aligned   DataFrame (n_cells × n_cell_types, F_obs subset)
        y_true          list[str]
        y_pred          list[str]
    """
    if cell_types is None:
        cell_types = sorted(F_obs.columns.tolist())

    # ── Align start cells to F_obs ────────────────────────────────────────
    ids_str  = np.array([str(i) for i in start_cell_ids])
    fobs_str = F_obs.index.astype(str)
    in_fobs  = pd.Series(ids_str).isin(fobs_str).values

    n_dropped = int((~in_fobs).sum())
    if n_dropped:
        log.warning(
            f"  {n_dropped}/{len(ids_str)} start cells not found in F_obs "
            "— dropped."
        )
    if not in_fobs.any():
        raise ValueError(
            "None of start_cell_ids found in F_obs.index.\n"
            f"  start_cell_ids sample : {ids_str[:5].tolist()}\n"
            f"  F_obs.index sample    : {list(fobs_str[:5])}"
        )

    cells_filt = start_cells[in_fobs]
    ids_filt   = ids_str[in_fobs]
    try:
        F_obs_aln  = F_obs.loc[ids_filt].reindex(columns=cell_types, fill_value=0.0)
    except KeyError:
        F_obs.index = ids_str
        F_obs_aln  = F_obs.loc[ids_filt].reindex(columns=cell_types, fill_value=0.0)
    log.info(f"Start cells matched to F_obs: {len(cells_filt)}")

    # ── KNN on reference data (skip fit if caller passed a pre-fit knn) ───
    if knn is None:
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        knn.fit(x_ref, y_ref)
        log.info(f"KNN fitted: {len(x_ref)} ref cells, k={k}")
    else:
        log.info(f"KNN reused (pre-fit, k={knn.n_neighbors})")

    # ── Predict F_hat ─────────────────────────────────────────────────────
    F_hat_arr = predict_fates_per_cell(
        start_cells     = cells_filt,
        model           = model,
        simulation_func = simulation_func,
        knn             = knn,
        cell_types      = cell_types,
        n_sims          = n_sims,
        device          = device,
    )
    F_hat = pd.DataFrame(F_hat_arr, index=ids_filt, columns=cell_types)

    # ── Compute accuracy ──────────────────────────────────────────────────
    result = compute_fate_accuracy(F_obs_aln, F_hat)
    result.update(
        n_start_cells = len(cells_filt),
        F_hat         = F_hat,
        F_obs_aligned = F_obs_aln,
    )
    log.info(
        f"Accuracy: {result['accuracy']:.4f}   "
        f"Pearson r: {result['pearson_r']:.4f}"
    )
    return result


# ─── simulation_func factories ────────────────────────────────────────────────

def make_prescient_sim_fn(num_steps: int = None):
    """
    Return a simulation_func for PRESCIENT.

    Usage
    -----
        model = (net, config)   # net: AutoGenerator, config: SimpleNamespace
        sim_fn = make_prescient_sim_fn()
        # or override steps: make_prescient_sim_fn(num_steps=20)

        endpoints = sim_fn(start_cells, model, n_sims=100, device="cuda:0")
        # returns ndarray (n_cells, n_sims, n_dims)

    Notes
    -----
    All n_cells × n_sims copies are batched into one forward pass for
    efficiency.  num_steps is auto-calculated from config if not given:
        (config.train_t[-1] - config.start_t) / config.train_dt
    """
    import torch

    def _simulate(start_cells, model, n_sims, device):
        net, config = model
        steps = num_steps or int(
            (config.train_t[-1] - config.start_t) / config.train_dt
        )
        n_cells, n_dims = start_cells.shape

        # Expand each cell n_sims times → (n_cells * n_sims, n_dims)
        x = torch.tensor(
            np.repeat(start_cells, n_sims, axis=0), dtype=torch.float32
        ).to(device)

        net.eval()
        # with torch.no_grad():
        for _ in range(steps):
            z = torch.randn_like(x) * config.train_sd
            x = net._step(x, dt=config.train_dt, z=z)

        return x.detach().cpu().numpy().reshape(n_cells, n_sims, n_dims)

    return _simulate


def make_tigon_sim_fn(t_start: float, t_end: float):
    """
    Return a simulation_func for TIGON.

    Usage
    -----
        model = func           # TorchDiffEqPack UOT model
        sim_fn = make_tigon_sim_fn(t_start=0.0, t_end=1.0)

        endpoints = sim_fn(start_cells, model, n_sims=1, device="cuda:0")
        # returns ndarray (n_cells, n_dims)  — deterministic ODE

    Notes
    -----
    TIGON is a deterministic ODE so n_sims is ignored (returns one endpoint
    per start cell).  Run run_fate_evaluation with n_sims=1 or any value;
    the pipeline handles the (n_cells, n_dims) output automatically.
    """
    import torch
    from TorchDiffEqPack import odesolve

    def _simulate(start_cells, model, n_sims, device):
        func = model
        z = torch.tensor(start_cells, dtype=torch.float32,
                         requires_grad=True).to(device)
        g0    = torch.zeros(z.shape[0], 1, dtype=torch.float32, device=device)
        logp0 = torch.zeros(z.shape[0], 1, dtype=torch.float32, device=device)

        options = {
            "method": "Dopri5", "h": None,
            "rtol": 1e-3, "atol": 1e-5,
            "print_neval": False, "neval_max": 1_000_000,
            "safety": None, "t0": t_start, "t1": t_end,
        }
        with torch.no_grad():
            z_out, _, _ = odesolve(func, y0=(z, g0, logp0), options=options)

        return z_out.detach().cpu().numpy()   # (n_cells, n_dims)

    return _simulate


def make_pseudodynamics_sim_fn(
    t_start_norm: float,
    t_end_norm: float,
    tol: float = 1e-5,
):
    """
    Return a simulation_func for pseudodynamics+.

    Usage
    -----
        model = pde_model      # pseudodynamics PDE model with
                               #   .v(state, t_in) and .time_scale_factor
        sim_fn = make_pseudodynamics_sim_fn(t_start_norm=0.0, t_end_norm=4.0)

        endpoints = sim_fn(start_cells, model, n_sims=1, device="cuda:0")
        # returns ndarray (n_cells, n_dims)  — deterministic ODE

    Notes
    -----
    Pseudodynamics+ uses a deterministic ODE; n_sims is ignored.

    t_start_norm / t_end_norm are in the *human-readable* normalised time
    domain (e.g. for Klein with norm_time='min_minus': tp2=0, tp4=2, tp6=4).
    Division by time_scale_factor is handled internally, matching the
    convention in pseudodynamics_plus/scripts/klein_eval.py::propagate().
    """
    import torch
    from torchdiffeq import odeint

    def _simulate(start_cells, model, n_sims, device):
        pde_model = model
        tsf = pde_model.time_scale_factor

        def v_ode(t, state):
            t_in = torch.full(
                (state.shape[0], 1), t.item() * tsf, dtype=torch.float32
            ).to(device)
            return pde_model.v(state, t_in)

        s0     = torch.from_numpy(start_cells.astype(np.float32)).to(device)
        t_eval = torch.tensor(
            [t_start_norm / tsf, t_end_norm / tsf], dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            traj = odeint(v_ode, s0, t_eval, method="dopri5", atol=tol, rtol=tol)

        return traj[-1].cpu().numpy()   # (n_cells, n_dims)

    return _simulate


def make_pseudodynamics_sde_sim_fn(
    t_start_norm: float,
    t_end_norm: float,
    n_steps: int = 200,
    noise_scale: float = 1.0,
):
    """
    Return a *stochastic* simulation_func for pseudodynamics+.

    Uses Euler-Maruyama integration with the learned velocity field v(s,t)
    and diffusion field D(s,t):

        dX = v(X,t)·dt + sqrt(2·|D(X,t)|·dt) · noise_scale · dW

    Running n_sims independent trajectories per cell produces diverse
    endpoints, giving a non-degenerate fate distribution F_hat.

    Usage
    -----
        sim_fn = make_pseudodynamics_sde_sim_fn(
            t_start_norm=0.0, t_end_norm=4.0, noise_scale=1.0,
        )
        endpoints = sim_fn(start_cells, pde_model, n_sims=100, device="cuda:0")
        # returns ndarray (n_cells, n_sims, n_dims)

    Parameters
    ----------
    t_start_norm / t_end_norm : float
        Normalised time domain (same convention as the deterministic sim).
    n_steps : int
        Euler-Maruyama discretisation steps.
    noise_scale : float
        Global multiplier on the stochastic term.  Set to 0 for the
        deterministic limit; sweep [0.1, 0.5, 1.0, 2.0] to tune.
    """
    import torch

    def _simulate(start_cells, model, n_sims, device):
        pde_model = model
        tsf = pde_model.time_scale_factor

        t0 = t_start_norm / tsf
        t1 = t_end_norm / tsf
        dt = (t1 - t0) / n_steps

        n_cells, n_dims = start_cells.shape
        sqrt_dt = np.sqrt(abs(dt))

        # Expand each cell n_sims times → (n_cells * n_sims, n_dims)
        s0 = torch.from_numpy(
            np.repeat(start_cells.astype(np.float32), n_sims, axis=0)
        ).to(device)

        pde_model.eval()
        s = s0.clone()
        with torch.no_grad():
            for step in range(n_steps):
                t_val = t0 + step * dt
                t_in = torch.full(
                    (s.shape[0], 1), t_val * tsf, dtype=torch.float32
                ).to(device)

                v = pde_model.v(s, t_in)                   # (N, n_dims)
                D = pde_model.D(s, t_in)                   # (N, 1) or (N, n_dims)

                # Euler-Maruyama step
                dW = torch.randn_like(s) * sqrt_dt         # (N, n_dims)
                diffusion_coeff = torch.sqrt(
                    2.0 * torch.abs(D)
                )                                           # (N, 1 or n_dims)
                # Broadcast D to match s shape if collapsed
                s = s + v * dt + noise_scale * diffusion_coeff.unsqueeze(dim=1) * dW

        endpoints = s.cpu().numpy().reshape(n_cells, n_sims, n_dims)
        return endpoints

    return _simulate


def make_pseudodynamics_sb_sim_fn(
    t_start_norm: float,
    t_end_norm: float,
    n_steps: int = 200,
    noise_scale: float = 1.0,
):
    """
    Return a score-guided SDE (Schrödinger Bridge) simulation_func.

    Augments the velocity field with the score function ∇_s log u(s,t)
    computed via autograd on the density surrogate:

        dX = [v(X,t) + noise_scale² · ∇_s log u(X,t)] dt
             + noise_scale · sqrt(2) · dW

    This concentrates trajectories in high-density regions while allowing
    stochastic branching — the Schrödinger Bridge formulation.

    Usage
    -----
        sim_fn = make_pseudodynamics_sb_sim_fn(
            t_start_norm=0.0, t_end_norm=4.0, noise_scale=1.0,
        )
        endpoints = sim_fn(start_cells, pde_model, n_sims=100, device="cuda:0")
        # returns ndarray (n_cells, n_sims, n_dims)

    Parameters
    ----------
    t_start_norm / t_end_norm : float
        Normalised time domain.
    n_steps : int
        Euler-Maruyama discretisation steps.
    noise_scale : float
        Controls diffusion magnitude.  The score drift scales as σ² and
        the noise scales as σ, so larger values give more stochasticity
        while keeping the SB balance.
    """
    import torch

    def _simulate(start_cells, model, n_sims, device):
        pde_model = model
        tsf = pde_model.time_scale_factor

        t0 = t_start_norm / tsf
        t1 = t_end_norm / tsf
        dt = (t1 - t0) / n_steps
        sigma2 = noise_scale ** 2
        sqrt_dt = np.sqrt(abs(dt))

        n_cells, n_dims = start_cells.shape

        # Expand each cell n_sims times → (n_cells * n_sims, n_dims)
        s0 = torch.from_numpy(
            np.repeat(start_cells.astype(np.float32), n_sims, axis=0)
        ).to(device)

        pde_model.eval()
        s = s0.clone()

        # Score computation requires grad; we enable it selectively
        for step in range(n_steps):
            t_val = t0 + step * dt

            # -- compute score ∇_s log u(s,t) with autograd --
            s_grad = s.detach().requires_grad_(True)
            t_in = torch.full(
                (s_grad.shape[0], 1), t_val * tsf, dtype=torch.float32
            ).to(device)
            t_in.requires_grad_(False)

            log_u = pde_model.u(s_grad, t_in)     # log u(s,t)
            score = torch.autograd.grad(
                log_u.sum(), s_grad, create_graph=False
            )[0]                                   # ∇_s log u

            with torch.no_grad():
                v = pde_model.v(s, t_in)           # (N, n_dims)
                dW = torch.randn_like(s) * sqrt_dt

                # SB drift: v + σ² ∇log u
                drift = v + sigma2 * score.detach()
                s = s + drift * dt + noise_scale * np.sqrt(2.0) * dW

        endpoints = s.detach().cpu().numpy().reshape(n_cells, n_sims, n_dims)
        return endpoints

    return _simulate


def make_trajectorynet_sim_fn(int_tp_start: float, int_tp_end: float):
    """
    Return a simulation_func for TrajectoryNet.

    Usage
    -----
        model = cnf_model      # TrajectoryNet SequentialFlow
        sim_fn = make_trajectorynet_sim_fn(int_tp_start=1.0, int_tp_end=2.0)

        endpoints = sim_fn(start_cells, model, n_sims=1, device="cuda:0")
        # returns ndarray (n_cells, n_dims)  — deterministic CNF

    Notes
    -----
    TrajectoryNet's CNF is deterministic; n_sims is ignored.
    int_tp_start / int_tp_end come from tjn_args.int_tps.
    """
    import torch

    def _simulate(start_cells, model, n_sims, device):
        cnf_model = model
        zs = [torch.tensor(start_cells, dtype=torch.float32).to(device)]
        zero = torch.zeros(zs[0].shape[0], 1, device=device)
        # int_times = torch.tensor(
        #     [int_tp_start, int_tp_end], dtype=torch.float32
        # ).to(device)

        int_tps = np.linspace(int_tp_start, int_tp_end, 100)

        cnf = cnf_model.chain[0]
        for i, itp in enumerate(int_tps[1:]):
            # tp counts down from last
            timescale = int_tps[1] - int_tps[0]
            integration_times = torch.tensor([itp - timescale, itp])
            # integration_times = torch.tensor([np.linspace(itp - args.time_scale, itp, ntimes)])
            integration_times = integration_times.type(torch.float32).to(device)

            # transform to previous timepoint
            z, _ = cnf(zs[-1], zero, integration_times=integration_times, reverse=True)
            zs.append(z)

        return z.detach().cpu().numpy()   # (n_cells, n_dims)

    return _simulate


def make_otcfm_sim_fn(t_start: float = 0.0, t_end: float = 2.0,
                      solver: str = "dopri5", n_steps: int = 100):
    """
    Return a simulation_func for OT-CFM.

    Usage
    -----
        model = mlp_model      # torchcfm MLP (time-varying)
        sim_fn = make_otcfm_sim_fn(t_start=0.0, t_end=2.0)

        endpoints = sim_fn(start_cells, model, n_sims=1, device="cuda:0")
        # returns ndarray (n_cells, n_dims)  — deterministic ODE

    Notes
    -----
    OT-CFM is a deterministic ODE; n_sims is ignored.
    Uses torchdyn NeuralODE with torch_wrapper for compatibility.
    """
    import torch
    from torchdyn.core import NeuralODE
    from torchcfm.utils import torch_wrapper

    def _simulate(start_cells, model, n_sims, device):
        node = NeuralODE(torch_wrapper(model), solver=solver)
        x0 = torch.tensor(start_cells, dtype=torch.float32).to(device)
        with torch.no_grad():
            traj = node.trajectory(
                x0,
                t_span=torch.linspace(t_start, t_end, n_steps, device=device),
            )
        return traj[-1].cpu().numpy()   # (n_cells, n_dims)

    return _simulate


def make_sf2m_ode_sim_fn(t_start: float = 0.0, t_end: float = 2.0,
                         solver: str = "euler", n_steps: int = 100):
    """
    Return a deterministic simulation_func for SF2M (drift model only).

    Usage
    -----
        model = drift_model    # torchcfm MLP (time-varying)
        sim_fn = make_sf2m_ode_sim_fn(t_start=0.0, t_end=2.0)

        endpoints = sim_fn(start_cells, model, n_sims=1, device="cuda:0")
        # returns ndarray (n_cells, n_dims)  — deterministic ODE

    Notes
    -----
    Uses only the drift model (no score). Deterministic; n_sims ignored.
    Default solver is "euler" following the torchcfm SF2M convention.
    """
    import torch
    from torchdyn.core import NeuralODE
    from torchcfm.utils import torch_wrapper

    def _simulate(start_cells, model, n_sims, device):
        node = NeuralODE(torch_wrapper(model), solver=solver)
        x0 = torch.tensor(start_cells, dtype=torch.float32).to(device)
        with torch.no_grad():
            traj = node.trajectory(
                x0,
                t_span=torch.linspace(t_start, t_end, n_steps, device=device),
            )
        return traj[-1].cpu().numpy()   # (n_cells, n_dims)

    return _simulate


def make_sf2m_sde_sim_fn(t_start: float = 0.0, t_end: float = 2.0,
                         sigma: float = 0.05, n_steps: int = 400):
    """
    Return a stochastic simulation_func for SF2M (drift + score via torchsde).

    Usage
    -----
        model = (drift_model, score_model)   # both torchcfm MLP
        sim_fn = make_sf2m_sde_sim_fn(t_start=0.0, t_end=2.0, sigma=0.05)

        endpoints = sim_fn(start_cells, model, n_sims=100, device="cuda:0")
        # returns ndarray (n_cells, n_sims, n_dims)

    Notes
    -----
    Stochastic SDE integration using torchsde.sdeint.
    Each start cell is replicated n_sims times for independent trajectories.
    """
    import torch
    import torchsde

    class _SDE(torch.nn.Module):
        noise_type = "diagonal"
        sde_type = "ito"

        def __init__(self, drift, score, input_size, sig):
            super().__init__()
            self.drift = drift
            self.score = score
            self.input_size = input_size
            self.sigma = sig

        def f(self, t, y):
            y = y.view(-1, *self.input_size)
            if len(t.shape) == len(y.shape):
                x = torch.cat([y, t], 1)
            else:
                x = torch.cat([y, t.repeat(y.shape[0])[:, None]], 1)
            return self.drift(x).flatten(start_dim=1) + self.score(x).flatten(start_dim=1)

        def g(self, t, y):
            return torch.ones_like(y) * self.sigma

    def _simulate(start_cells, model, n_sims, device):
        drift_model, score_model = model
        n_cells, n_dims = start_cells.shape

        sde = _SDE(drift_model, score_model, input_size=(n_dims,), sig=sigma)

        # Expand each cell n_sims times → (n_cells * n_sims, n_dims)
        x0 = torch.from_numpy(
            np.repeat(start_cells.astype(np.float32), n_sims, axis=0)
        ).to(device)

        with torch.no_grad():
            traj = torchsde.sdeint(
                sde, x0,
                ts=torch.linspace(t_start, t_end, n_steps, device=device),
            )

        endpoints = traj[-1].cpu().numpy().reshape(n_cells, n_sims, n_dims)
        return endpoints

    return _simulate


def make_mioflow_sim_fn(t_start: float, t_end: float, n_steps: int = 100,
                        autoencoder=None, recon: bool = False):
    """
    Return a simulation_func for MIOFlow using generate_points().

    Usage
    -----
        model = toy_model      # MIOFlow ToyModel (wraps ToyODE + odeint)
        sim_fn = make_mioflow_sim_fn(t_start=2.0, t_end=6.0)

        endpoints = sim_fn(start_cells, model, n_sims=1, device="cuda:0")
        # returns ndarray (n_cells, n_dims) — deterministic ODE

    Parameters
    ----------
    autoencoder : MIOFlow Autoencoder or None
        If provided (with recon=True), generate_points encodes input,
        runs ODE in latent space, then decodes back to original space.
    recon : bool
        Whether to use encode/decode via the autoencoder.

    Notes
    -----
    Wraps MIOFlow's generate_points() with sample_time=np.linspace(t_start,
    t_end, n_steps). Builds a temporary DataFrame from start_cells so
    generate_points can consume it.
    """
    from MIOFlow.eval import generate_points

    def _simulate(start_cells, model, n_sims, device):
        n_cells, n_dims = start_cells.shape
        use_cuda = "cuda" in str(device) and torch.cuda.is_available()

        # Build df for generate_points (cells at t_start)
        df = pd.DataFrame(
            start_cells,
            columns=[f"d{i}" for i in range(1, n_dims + 1)],
        )
        df["samples"] = int(t_start)

        sample_time = np.linspace(t_start, t_end, n_steps)
        generated = generate_points(
            model, df, n_points=n_cells,
            sample_with_replacement=False,
            use_cuda=use_cuda,
            sample_time=sample_time,
            autoencoder=autoencoder,
            recon=recon,
        )
        # generated: (n_steps, n_cells, n_dims)
        return generated[-1]  # (n_cells, n_dims)

    return _simulate


# ─── Unstandardization helpers ───────────────────────────────────────────────

def inverse_standardize(x, scaler):
    """
    Inverse z-score standardization: x_original = x * std + mean.

    Parameters
    ----------
    x      : ndarray (n, d)
    scaler : dict-like with keys 'mean' and 'std', each ndarray (d,).
             If None, returns x unchanged.

    Returns
    -------
    ndarray (n, d)
    """
    if scaler is None:
        return x
    return x * np.asarray(scaler['std']) + np.asarray(scaler['mean'])


def compute_scaler_from_adata(adata, scaled_key, n_dims):
    """
    Compute scaler (mean, std) from adata by inferring the unscaled key.

    For pseudodynamics+, the adata contains both 'X_pca_scaled' and 'X_pca'
    (or 'DM_EigenVectors_scaled' and 'DM_EigenVectors').  The scaler is
    computed as mean/std of the unscaled version over ALL cells.

    Parameters
    ----------
    adata      : AnnData — FULL adata (not just test subset).
    scaled_key : str — e.g. 'X_pca_scaled', 'DM_EigenVectors_scaled'.
    n_dims     : int

    Returns
    -------
    dict with 'mean' and 'std' (each ndarray of shape (n_dims,)),
    or None if the key does not end with '_scaled'.
    """
    if not scaled_key.endswith('_scaled'):
        return None
    raw_key = scaled_key.replace('_scaled', '')
    if raw_key not in adata.obsm:
        raise KeyError(
            f"Expected unscaled key '{raw_key}' in adata.obsm but not found. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
    raw = adata.obsm[raw_key][:, :n_dims].astype(np.float64)
    mean = raw.mean(axis=0)
    std = np.clip(raw.std(axis=0), 1e-6, None)
    return {'mean': mean, 'std': std}


# ─── Method-agnostic W2 helpers ─────────────────────────────────────────────

def compute_w2(x_sim, x_true, device: str = "cpu"):
    """
    Compute exact W2 distance using POT library.

    Parameters
    ----------
    x_sim  : ndarray (n_sim, n_dims)
    x_true : ndarray (n_true, n_dims)

    Returns
    -------
    float : W2 distance
    """
    import ot
    from torchcfm.optimal_transport import wasserstein
    # x_s = np.asarray(x_sim, dtype=np.float64)
    # x_t = np.asarray(x_true, dtype=np.float64)
    # n_s, n_t = x_s.shape[0], x_t.shape[0]
    # w_a = np.ones(n_s) / n_s
    # w_b = np.ones(n_t) / n_t
    # M = ot.dist(x_s, x_t, metric='sqeuclidean')
    # w2_sq = ot.emd2(w_a, w_b, M)

    # return float(np.sqrt(max(w2_sq, 0.0)))
    with torch.no_grad():
        w2 = wasserstein(torch.from_numpy(x_sim.astype(np.float32)).to(device), 
                     torch.from_numpy(x_true.astype(np.float32)).to(device),
                     reg=0.05)
    
    return w2


def eval_w2_tasks(eval_tasks, simulate_fn, n_sims=10, n_sim_cells=5000, scaler=None):
    """
    Evaluate W2 distance over a list of tasks.

    Parameters
    ----------
    eval_tasks : list of dict
        Each dict has keys: name, t_start, t_end, src_emb, target_emb.
    simulate_fn : callable(x_start, t_start, t_end) -> ndarray (n_cells, n_dims)
        Deterministic forward simulation.
    n_sims : int
        Number of replicates (with random subsampling).
    n_sim_cells : int
        Max cells per replicate for W2 computation.
    scaler : dict-like with 'mean'/'std' or None
        If provided, simulated and target embeddings are inverse-standardized
        before computing W2.

    Returns
    -------
    list of dict : [{task, replicate, w2}, ...]
    """
    results = []
    for task in eval_tasks:
        tgt_orig = inverse_standardize(task['target_emb'], scaler)
        for rep in range(n_sims):
            src = task['src_emb']
            if src.shape[0] > n_sim_cells:
                idx = np.random.choice(src.shape[0], n_sim_cells, replace=False)
                src = src[idx]

            z_sim = simulate_fn(src, task['t_start'], task['t_end'])
            z_sim = inverse_standardize(z_sim, scaler)

            tgt = tgt_orig
            if tgt.shape[0] > n_sim_cells:
                tgt = tgt[np.random.choice(tgt.shape[0], n_sim_cells, replace=False)]
            if z_sim.shape[0] > n_sim_cells:
                z_sim = z_sim[np.random.choice(z_sim.shape[0], n_sim_cells, replace=False)]

            w2 = compute_w2(z_sim, tgt)
            results.append({"task": task['name'], "replicate": rep + 1, "w2": w2})
    return results


# ─── Method-agnostic population & per-clone W2 ──────────────────────────────

def compute_w2_population(
    src_cells: np.ndarray,
    target_cells: np.ndarray,
    model,
    sim_fn: callable,
    n_sims: int = 10,
    device: str = "cpu",
    scaler=None,
):
    """
    Compute population-level W2 in both standardized and raw space.

    Parameters
    ----------
    src_cells    : ndarray (n_src, n_dims) — source cells to simulate forward.
    target_cells : ndarray (n_tgt, n_dims) — observed target cells.
    model        : model object, passed to sim_fn.
    sim_fn       : callable(start_cells, model, n_sims, device) -> ndarray.
    n_sims       : trajectories per cell (for stochastic; deterministic ignores).
    device       : torch device string.
    scaler       : dict with 'mean'/'std' or None.

    Returns
    -------
    dict with keys 'w2_scaled' and 'w2_raw'.
    """
    endpoints = np.asarray(sim_fn(src_cells, model, n_sims, device))
    # Flatten stochastic: (n_cells, n_sims, n_dims) → (n_cells*n_sims, n_dims)
    if endpoints.ndim == 3:
        endpoints = endpoints.reshape(-1, endpoints.shape[-1])

    w2_scaled = compute_w2(endpoints, target_cells)

    if scaler is not None:
        ep_raw = inverse_standardize(endpoints, scaler)
        tgt_raw = inverse_standardize(target_cells, scaler)
        w2_raw = compute_w2(ep_raw, tgt_raw)
    else:
        w2_raw = w2_scaled

    return {"w2_scaled": w2_scaled, "w2_raw": w2_raw}


def compute_w2_per_clone(
    src_cells: np.ndarray,
    src_clone_ids: np.ndarray,
    target_cells: np.ndarray,
    target_clone_ids: np.ndarray,
    model,
    sim_fn: callable,
    n_sims: int = 10,
    device: str = "cpu",
    scaler=None,
):
    """
    Compute per-clone W2 distance in both standardized and raw space.

    Parameters
    ----------
    src_cells        : ndarray (n_src, n_dims)
    src_clone_ids    : array-like (n_src,) — clone ID per source cell.
    target_cells     : ndarray (n_tgt, n_dims)
    target_clone_ids : array-like (n_tgt,) — clone ID per target cell.
    model, sim_fn, n_sims, device, scaler : see compute_w2_population.

    Returns
    -------
    DataFrame with columns: clone, w2_scaled, w2_raw, clone_size_src, clone_size_tgt.
    """
    src_ids = np.asarray(src_clone_ids)
    tgt_ids = np.asarray(target_clone_ids)
    shared_clones = sorted(set(src_ids) & set(tgt_ids))

    records = []
    for clone_id in shared_clones:
        s_mask = src_ids == clone_id
        t_mask = tgt_ids == clone_id
        if s_mask.sum() == 0 or t_mask.sum() == 0:
            continue

        sc_cells = src_cells[s_mask]
        tc_cells = target_cells[t_mask]

        endpoints = np.asarray(sim_fn(sc_cells, model, n_sims, device))
        if endpoints.ndim == 3:
            endpoints = endpoints.reshape(-1, endpoints.shape[-1])

        w2_scaled = compute_w2(endpoints, tc_cells)

        if scaler is not None:
            ep_raw = inverse_standardize(endpoints, scaler)
            tc_raw = inverse_standardize(tc_cells, scaler)
            w2_raw = compute_w2(ep_raw, tc_raw)
        else:
            w2_raw = w2_scaled

        records.append({
            "clone": clone_id,
            "w2_scaled": w2_scaled,
            "w2_raw": w2_raw,
            "clone_size_src": int(s_mask.sum()),
            "clone_size_tgt": int(t_mask.sum()),
        })

    return pd.DataFrame(records)


# ─── Legacy pseudodynamics-specific W2 helpers ──────────────────────────────

def klein_w2_v1(test_ad: sc.AnnData,
                clone_proportions: pd.DataFrame,
                config,
                device: str,
                sim_fn: callable,
                n_sims: int = 10,
                scaler=None):
    R"""
    Population-level W2 from t4 → t6 over testset clones.

    Returns
    -------
    dict with keys 'w2_scaled' and 'w2_raw'.
    Both computed via `compute_w2` (entropic OT, reg=0.05) — same backend
    as PRESCIENT / sf2m / otcfm. `w2_raw` requires `scaler`; otherwise it
    equals `w2_scaled`.
    """
    import pseudodynamics as pdp

    n_dims = config.dataset_config['n_dimension']
    cellstate_key = config.dataset_config.get("cellstate_key", None)

    w2_clone_mask = test_ad.obs.clones.isin(clone_proportions.index)
    w2_test_ad = test_ad[w2_clone_mask].copy()
    start_ad = w2_test_ad[w2_test_ad.obs.timepoint_tx_days == 4]
    ref_ad = w2_test_ad[w2_test_ad.obs.timepoint_tx_days == 6]

    start_cells = start_ad.obsm[cellstate_key][:, :n_dims]
    ref_cells = ref_ad.obsm[cellstate_key][:, :n_dims]

    pde_model = pdp.models.pde_params.load_from_checkpoint(
        config.find_lastest_ckpt()).to(device)

    endpoints = np.asarray(sim_fn(start_cells, pde_model, n_sims, device))
    endpoints = endpoints.reshape(-1, start_cells.shape[1])

    w2_scaled = compute_w2(endpoints, ref_cells, device=device)
    if scaler is not None:
        ep_raw = inverse_standardize(endpoints, scaler)
        rc_raw = inverse_standardize(ref_cells, scaler)
        w2_raw = compute_w2(ep_raw, rc_raw, device=device)
    else:
        w2_raw = w2_scaled

    return {"w2_scaled": float(w2_scaled), "w2_raw": float(w2_raw)}


def klein_w2_v2(test_ad: sc.AnnData,
                clone_proportions: pd.DataFrame,
                config,
                device: str,
                sim_fn: callable,
                n_sims: int = 10,
                scaler=None):
    R"""
    Per-clone W2 from t4 → t6 over testset clones.

    Returns
    -------
    DataFrame with columns: clone, w2_scaled, w2_raw, clone_size_t4, clone_size_t6.
    Same backend (`compute_w2`, reg=0.05) as PRESCIENT / sf2m / otcfm.
    """
    import pseudodynamics as pdp

    n_dims = config.dataset_config['n_dimension']
    cellstate_key = config.dataset_config.get("cellstate_key", None)

    w2_clone_mask = test_ad.obs.clones.isin(clone_proportions.index)
    w2_test_ad = test_ad[w2_clone_mask].copy()

    pde_model = pdp.models.pde_params.load_from_checkpoint(
        config.find_lastest_ckpt()).to(device)

    start_ad = w2_test_ad[w2_test_ad.obs.timepoint_tx_days == 4]
    ref_ad = w2_test_ad[w2_test_ad.obs.timepoint_tx_days == 6]

    ref_all_scaled = ref_ad.obsm[cellstate_key][:, :n_dims]
    ref_all_raw = inverse_standardize(ref_all_scaled, scaler) if scaler is not None else ref_all_scaled

    clones = start_ad.obs.clones.unique()
    records = []
    for clone_id in clones:
        cmask_start = start_ad.obs.clones == clone_id
        cmask_ref = ref_ad.obs.clones == clone_id
        if cmask_start.sum() == 0 or cmask_ref.sum() == 0:
            continue

        sc_cells = start_ad[cmask_start].obsm[cellstate_key][:, :n_dims]
        rc_scaled = ref_all_scaled[cmask_ref.values]
        rc_raw = ref_all_raw[cmask_ref.values]

        endpoints = np.asarray(sim_fn(sc_cells, pde_model, n_sims, device))
        endpoints = endpoints.reshape(-1, sc_cells.shape[1])

        w2_scaled = compute_w2(endpoints, rc_scaled, device=device)
        if scaler is not None:
            ep_raw = inverse_standardize(endpoints, scaler)
            w2_raw = compute_w2(ep_raw, rc_raw, device=device)
        else:
            w2_raw = w2_scaled

        records.append({
            "clone": clone_id,
            "w2_scaled": float(w2_scaled),
            "w2_raw": float(w2_raw),
            "clone_size_t4": int(cmask_start.sum()),
            "clone_size_t6": int(cmask_ref.sum()),
        })

    return pd.DataFrame(records)