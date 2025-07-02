# Physics-Informed Synthetic Data Generation for Cellular Dynamics

## Data Synthesis Framework

### Governing Equations
The synthetic data generation follows a partial differential equation (PDE) framework that incorporates growth, drift, and diffusion dynamics:

$$\frac{\partial u}{\partial t} = \underbrace{g(\mathbf{s})u}_{\text{Growth}} - \underbrace{\nabla\cdot(\mathbf{v}(\mathbf{s})u)}_{\text{Drift}} + \underbrace{\nabla\cdot(D(\mathbf{s})\nabla u)}_{\text{Diffusion}}$$

where:
- $u(\mathbf{s},t)$: cell density at position $\mathbf{s}$ and time $t$
- $g(\mathbf{s})$: growth rate function
- $\mathbf{v}(\mathbf{s})$: velocity field
- $D(\mathbf{s})$: diffusion coefficient

### Parameter Modeling
Parameters are modeled using multi-dimensional cubic splines:

$$g(\mathbf{s}) = \text{MultiDim_CubicSpline}(\mathbf{G}, \mathbf{X})$$
$$\mathbf{v}(\mathbf{s}) = \text{MultiDim_CubicSpline}(\mathbf{V}, \mathbf{X})$$
$$D(\mathbf{s}) = \text{MultiDim_CubicSpline}(\mathbf{D}, \mathbf{X})$$

### Simulation Pipeline

```python
# Pseudocode for synthetic data generation
def generate_synthetic_data():
    # 1. Parameter initialization
    x_knots, v_knots = generate_v_knots()
    g_knots = generate_g_knots()
    D_knots = generate_D_knots()
    
    # 2. Model construction
    g_cs = MultiDim_CubicSpline(g_knots.T, x_knots.T, collapse=True)
    v_cs = MultiDim_CubicSpline(v_knots.T, x_knots.T, collapse=True)
    D_cs = MultiDim_CubicSpline(D_knots.T, x_knots.T, collapse=True)
    
    # 3. ODE system definition
    model = Syn_model(g_cs, v_cs, D_cs)
    
    # 4. Initial condition setup
    u_init = compute_initial_density()
    cellstate = compute_initial_cell_states()
    du_dcs_t = compute_spatial_derivatives()
    
    # 5. Temporal evolution
    sim_out = odeint(model, (u_init, cellstate, du_dcs_t, ...),
                    t=torch.linspace(0, T, N_timepoints))
    
    # 6. Data storage
    save_results(sim_out, params)
```

## Model Evaluation Framework

### Network Architecture
The physics-informed neural network follows this architecture:

$$\begin{aligned}
&\text{Input: } \mathbf{s}_i \in \mathbb{R}^d \\
&\text{Hidden Layers: } \text{MLP}(\text{dim}=[d+1, 16, 16, 1]) \\
&\text{Output: } \hat{g}(\mathbf{s}_i), \hat{\mathbf{v}}(\mathbf{s}_i), \hat{D}(\mathbf{s}_i)
\end{aligned}$$

### Training Protocol
Training follows these specifications:

```yaml
# Training Configuration
channels: [5+1, 16, 16, 1]
g_channels: [5, 16, 16, 1]
v_channels: [5, 32, 16, 5]
D_channels: [5, 32, 1]
activation_fn: 'Tanh'
ode_tol: 1e-4
D_penalty: 0.5
deltax_weight: 3
weight_intensity: 1
time_scale_factor: 1
time_sensitive: True
growth_weight: 3
```

### Evaluation Metrics
1. **Density Reconstruction Accuracy**:
   - KL-Divergence: $D_{KL}(p||q) = \sum_i p_i \log\frac{p_i}{q_i}$
   - Normalized MSE: $\text{NMSE} = \frac{\|\mathbf{u}_{\text{true}} - \mathbf{u}_{\text{pred}}\|^2}{\|\mathbf{u}_{\text{true}}\|^2}$

2. **Parameter Estimation Accuracy**:
   - Spearman's rank correlation: $\rho = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$
   - Pearson correlation: $r = \frac{\text{cov}(X,Y)}{\sigma_X\sigma_Y}$

3. **Time-sensitive Performance**:
   - Validation on leave-out timepoints
   - Test on unseen timepoints

### Implementation Details
1. **Data Splitting**:
   - Training: [0, 1, 2, 4, 6, 8, 10]
   - Validation: [5, 7]
   - Test: [3, 9]

2. **Optimization**:
   - Optimizer: PyTorch Lightning Trainer with GPU acceleration
   - Learning rate: Auto-tuned
   - Epochs: 300
   - Batch size: 512

3. **Regularization**:
   - Diffusion penalty: $\lambda_D = 0.5$
   - Delta-x weight: $\lambda_\Delta = 3$
   - Growth weight: $\lambda_g = 3$

### Reproducibility Checklist
1. **Initial Conditions**:
   - Starting cells: 2000 per timepoint
   - Initial distribution: Gaussian mixture
   - Dimensionality: 5D

2. **Simulation Parameters**:
   - Time integration: 0-4 days
   - Time resolution: 11 timepoints
   - Spatial resolution: 22,000 cells

3. **Implementation Requirements**:
   - Python 3.9+
   - PyTorch 1.13+
   - Scanpy 1.9+
   - Phate 0.8+
   - UMAP 0.5+