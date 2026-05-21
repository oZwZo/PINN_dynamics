# torchcfm Package Reference

## Installation

```bash
pip install torchcfm
# Dependencies: torch, torchdyn, pot (Python Optimal Transport)
# Optional for SDE: torchsde
```

## Core Classes

### ConditionalFlowMatcher Classes

All located in `torchcfm.conditional_flow_matching`:

| Class | Description | sigma meaning |
|-------|-------------|---------------|
| `ConditionalFlowMatcher(sigma)` | Vanilla CFM | Gaussian path noise |
| `ExactOptimalTransportConditionalFlowMatcher(sigma)` | OT-CFM with exact OT coupling | Same |
| `SchrodingerBridgeConditionalFlowMatcher(sigma)` | SB-CFM / SF2M flow matcher | Diffusion coefficient |
| `TargetConditionalFlowMatcher(sigma)` | Target-conditioned variant | Same |

**Common API** (all classes):
```python
FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.1)

# Core sampling method
t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
# With noise (needed for SF2M score loss):
t, xt, ut, eps = FM.sample_location_and_conditional_flow(x0, x1, return_noise=True)

# SF2M-specific: compute lambda weighting for score loss
lambda_t = SF2M.compute_lambda(t)  # t should be in [0,1]
```

**Parameters:**
- `x0`: (batch, dim) source samples
- `x1`: (batch, dim) target samples
- `t`: (batch,) random times in [0,1]
- `xt`: (batch, dim) interpolated points
- `ut`: (batch, dim) conditional flow (target for regression)
- `eps`: (batch, dim) noise (for score matching)

### OTPlanSampler

```python
from torchcfm.optimal_transport import OTPlanSampler
ot_sampler = OTPlanSampler(method="exact")  # or "sinkhorn"
pi = ot_sampler.get_map(x0, x1)  # OT coupling matrix
x0_paired, x1_paired = ot_sampler.sample_map(pi, batch_size)
```

### Wasserstein Distance

```python
from torchcfm.optimal_transport import wasserstein
w2 = wasserstein(x_sim, x_true, reg=0.05)  # Sinkhorn-regularized W2
```

## Neural Network: MLP

```python
from torchcfm.models import MLP

model = MLP(dim=30, time_varying=True, w=128)
# Architecture: Linear(dim+1, w) -> SELU -> Linear(w, w) -> SELU -> Linear(w, dim)
# Input: cat([x, t], dim=-1) -> shape (batch, dim+1)
# Output: shape (batch, dim)

vt = model(torch.cat([xt, t[:, None]], dim=-1))
```

## Model Wrapper for torchdyn

```python
from torchcfm.utils import torch_wrapper

# Wraps model for NeuralODE compatibility (handles time tensor format)
wrapped = torch_wrapper(model)
```

## Training Pattern

### OT-CFM Training
```python
FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.1)
model = MLP(dim=dim, time_varying=True, w=128)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for i in range(n_iters):
    optimizer.zero_grad()
    t, xt, ut = get_batch(FM, X, batch_size, n_times)
    vt = model(torch.cat([xt, t[:, None]], dim=-1))
    loss = torch.mean((vt - ut) ** 2)
    loss.backward()
    optimizer.step()
```

### SF2M Training (dual model: drift + score)
```python
SF2M = SchrodingerBridgeConditionalFlowMatcher(sigma=0.05)
drift_model = MLP(dim=dim, time_varying=True, w=128)
score_model = MLP(dim=dim, time_varying=True, w=128)
optimizer = torch.optim.AdamW(
    list(drift_model.parameters()) + list(score_model.parameters()), lr=1e-4
)

for i in range(n_iters):
    optimizer.zero_grad()
    t, xt, ut, eps = get_batch(SF2M, X, batch_size, n_times, return_noise=True)
    lambda_t = SF2M.compute_lambda(t % 1)
    vt = drift_model(torch.cat([xt, t[:, None]], dim=-1))
    st = score_model(torch.cat([xt, t[:, None]], dim=-1))
    flow_loss = torch.mean((vt - ut) ** 2)
    score_loss = torch.mean((lambda_t[:, None] * st + eps) ** 2)
    loss = flow_loss + score_loss
    loss.backward()
    optimizer.step()
```

### Batch Construction (multi-timepoint)
```python
def get_batch(FM, X, batch_size, n_times, return_noise=False):
    """Build batch from all consecutive timepoint pairs.

    X: list of arrays [X_tp0, X_tp1, ...], each (n_cells, dim)
    Time is offset by pair index so t ranges over [0, n_times-1].
    """
    ts, xts, uts, noises = [], [], [], []
    for t_start in range(n_times - 1):
        x0 = torch.from_numpy(
            X[t_start][np.random.randint(X[t_start].shape[0], size=batch_size)]
        ).float().to(device)
        x1 = torch.from_numpy(
            X[t_start + 1][np.random.randint(X[t_start + 1].shape[0], size=batch_size)]
        ).float().to(device)
        if return_noise:
            t, xt, ut, eps = FM.sample_location_and_conditional_flow(x0, x1, return_noise=True)
            noises.append(eps)
        else:
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        ts.append(t + t_start)
        xts.append(xt)
        uts.append(ut)
    t = torch.cat(ts)
    xt = torch.cat(xts)
    ut = torch.cat(uts)
    if return_noise:
        return t, xt, ut, torch.cat(noises)
    return t, xt, ut
```

## Sampling / Generation

### Deterministic ODE (OT-CFM or SF2M drift-only)
```python
from torchdyn.core import NeuralODE
from torchcfm.utils import torch_wrapper

node = NeuralODE(torch_wrapper(model), solver="dopri5")
# or solver="euler" for SF2M

traj = node.trajectory(
    x0,  # (n_cells, dim)
    t_span=torch.linspace(0, n_times - 1, n_steps),
)
# Returns (n_steps, n_cells, dim)
# Final state: traj[-1]  -> (n_cells, dim)
```

### Stochastic SDE (SF2M with score)
```python
import torchsde

class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, drift, score, input_size, sigma=0.05):
        super().__init__()
        self.drift = drift
        self.score = score
        self.input_size = input_size
        self.sigma = sigma

    def f(self, t, y):
        y = y.view(-1, *self.input_size)
        x = torch.cat([y, t.repeat(y.shape[0])[:, None]], 1)
        return self.drift(x).flatten(start_dim=1) + self.score(x).flatten(start_dim=1)

    def g(self, t, y):
        return torch.ones_like(y) * self.sigma

sde = SDE(drift_model, score_model, input_size=(dim,), sigma=0.05)
sde_traj = torchsde.sdeint(
    sde, x0,
    ts=torch.linspace(0, n_times - 1, n_steps),
)
# Returns (n_steps, n_cells, dim)
```

## OT-CFM vs SF2M Comparison

| Feature | OT-CFM | SF2M |
|---------|--------|------|
| Flow matcher | `ExactOptimalTransportConditionalFlowMatcher` | `SchrodingerBridgeConditionalFlowMatcher` |
| Models | 1 (velocity) | 2 (velocity + score) |
| Sigma | 0.1 (typical) | 0.05 (typical) |
| Optimizer | Adam | AdamW |
| Loss | MSE on flow | MSE on flow + weighted score matching |
| ODE solver | dopri5 | euler |
| Stochastic mode | No | Yes (SDE with torchsde) |
| `return_noise` | No | Yes (needed for score loss) |
| Output | Deterministic trajectories | Deterministic (ODE) or stochastic (SDE) |
