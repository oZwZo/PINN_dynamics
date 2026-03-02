# TrajectoryNet Reference

## Overview
TrajectoryNet is a Continuous Normalizing Flow (CNF) model for inferring cell trajectories from time-series single-cell data. It learns a continuous ODE that maps cells between observed timepoints, using neural ODEs as the velocity field.

- **Paper**: Tong et al., "TrajectoryNet: A Dynamic Optimal Transport Network for Modeling Cellular Dynamics" (ICML 2020)
- **Package**: `TrajectoryNet` (installed in `mioflow` conda env)
- **Source**: `/rds/user/wz369/hpc-work/LIBS/mamba/envs/mioflow/lib/python3.10/site-packages/TrajectoryNet/`

## Key Concepts
- Learns a **single ODE** that maps a base (Gaussian) density through time to match observed cell distributions
- Training uses **reverse integration**: starts from the last timepoint, integrates backwards, and accumulates log-likelihood losses
- Uses `torchdiffeq` for ODE solving (dopri5 by default)
- Supports optional growth models and velocity regularization

## CLI Arguments (from `TrajectoryNet/parse.py`)

### Data
- `--dataset`: Path to `.npz` or `.h5ad` file, or built-in name (EB, CIRCLE3, etc.)
- `--embedding_name`: Key in NPZ for embedding (default: `pca`)
- `--max_dim`: Clip embedding to this many dimensions (default: 10)
- `--whiten`: Standardize data (zero-mean, unit-variance) — skip if pre-standardized

### Model Architecture
- `--dims`: Hidden layer dimensions, e.g. `64-64-64` or `128-128-128`
- `--layer_type`: ODE layer type (default: `concatsquash`)
- `--nonlinearity`: Activation (default: `tanh`)
- `--num_blocks`: Number of stacked CNFs (default: 1)
- `--time_scale`: Scaling for integration times (default: 0.5)

### Training
- `--niters`: Training iterations (default: 10000)
- `--lr`: Learning rate (default: 1e-3)
- `--batch_size`: Cells per batch (default: 1000)
- `--training_noise`: Gaussian noise added to training data (default: 0.1)
- `--weight_decay`: L2 regularization (default: 1e-5)

### Growth & Regularization
- `--use_growth`: Enable growth model (action flag)
- `--vecint`: Velocity direction regularization weight
- `--interp_reg`: Straightline interpolation regularization

### Saving
- `--save`: Output directory for checkpoints and logs
- `--save_freq`: Checkpoint save frequency (default: 1000 iters)
- `--gpu`: GPU device index (default: 0)

## Data Format (NPZ for CustomData)

TrajectoryNet expects an NPZ file with:
- `<embedding_name>`: `(n_cells, n_dims)` float array — the cell embeddings
- `sample_labels`: `(n_cells,)` int array — timepoint labels (consecutive integers starting at 0)

**Critical**: Sample labels must be consecutive integers starting at 0. TrajectoryNet computes:
```python
int_tps = (np.arange(max(timepoints) + 1) + 1.0) * time_scale
```
With labels `[0, 1, 2]` and `time_scale=0.5`, this gives `int_tps = [0.5, 1.0, 1.5]` (3 integration points).
With labels `[2, 4, 6]`, this would give 7 integration points — wasteful and incorrect.

## Training Workflow

```bash
python -m TrajectoryNet.main \
    --dataset path/to/data.npz \
    --embedding_name X_pca \
    --max_dim 30 \
    --dims 128-128-128 \
    --niters 10000 \
    --save path/to/output \
    --gpu 0
```

### What happens during training:
1. Data loaded via `dataset.SCData.factory()` → `CustomData` for `.npz` files
2. Timepoints extracted: `args.timepoints = data.get_unique_times()`
3. Integration times computed: `args.int_tps = (np.arange(max(tp) + 1) + 1.0) * time_scale`
4. Model built via `build_model_tabular(args, dims, regularization_fns)`
5. Training loop: backward integration computing log-likelihood loss
6. Best model saved to `{save}/checkpt.pth` (based on validation loss)
7. Periodic checkpoints saved to `{save}/checkpt-{iter}.pth`

## Model Loading for Evaluation

```python
from TrajectoryNet.parse import parser as tjn_parser
from TrajectoryNet.train_misc import build_model_tabular, create_regularization_fns, set_cnf_options
from TrajectoryNet import dataset

# Reconstruct args
tjn_args = tjn_parser.parse_args([
    "--dataset", "path/to/data.npz",
    "--embedding_name", "X_pca",
    "--max_dim", "30",
    "--dims", "128-128-128",
    "--save", "path/to/model_dir",
])
data = dataset.SCData.factory(tjn_args.dataset, tjn_args)
tjn_args.data = data
tjn_args.timepoints = data.get_unique_times()
tjn_args.int_tps = (np.arange(max(tjn_args.timepoints) + 1) + 1.0) * tjn_args.time_scale

regularization_fns, _ = create_regularization_fns(tjn_args)
model = build_model_tabular(tjn_args, data.get_shape()[0], regularization_fns)
ckpt = torch.load("checkpt.pth", map_location=device)
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

## Forward Integration

The model's CNF integrates in a "time" space defined by `int_tps`. To simulate forward from timepoint `i` to `i+1`:

```python
cnf = model.chain[0]  # first (and usually only) CNF block
z_in = torch.tensor(source_cells, dtype=torch.float32).to(device)
zero = torch.zeros(z_in.shape[0], 1).to(device)

# int_tps[i] and int_tps[i+1] define the integration interval
integration_times = torch.tensor([int_tps[i], int_tps[i+1]]).float().to(device)

# reverse=False integrates forward in biological time
z_out, delta_logp = cnf(z_in, zero, integration_times=integration_times, reverse=False)
```

Note: `reverse=False` means forward in the model's integration direction, which corresponds to **forward** in biological time when using the `int_tps` ordering.

## Key Source Files

| File | Key Functions |
|------|--------------|
| `main.py` | `main()`, `train()`, `compute_loss()` — training entry point |
| `dataset.py` | `SCData.factory()`, `CustomData` — data loading |
| `train_misc.py` | `build_model_tabular()`, `create_regularization_fns()`, `set_cnf_options()` |
| `parse.py` | All CLI argument definitions |
| `lib/layers/__init__.py` | `CNF`, `SequentialFlow`, `ODEnet`, `ODEfunc` — model components |
