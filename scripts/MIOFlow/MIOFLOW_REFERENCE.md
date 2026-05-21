# MIOFlow Reference

MIOFlow (Manifold Interpolating Optimal-transport Flow) is a Neural ODE model trained via
optimal transport loss between consecutive timepoint distributions. Unlike potential-based
methods (PRESCIENT), it directly models the velocity field without assuming a potential energy
landscape.

**Paper**: Huguet et al., "Manifold Interpolating Optimal-Transport Flows for Trajectory
Inference", NeurIPS 2022.
**Repository**: https://github.com/KrishnaswamyLab/MIOFlow

## Installation

```bash
# Dedicated conda environment (already set up on HPC)
conda activate mioflow
# Python 3.10, installed at:
#   /home/wz369/rds/hpc-work/LIBS/mamba/envs/mioflow/
```

## API Reference (Old API)

The "old" API (`make_model`, `training_regimen`) is used in our scripts. The newer `MIOFlow`
class has incomplete `save()`/`load()` stubs and hardcodes PHATE assumptions, so we avoid it.

### `make_model(feature_dims, layers, activation, scales, use_cuda)`

Creates a `ToyModel` wrapping a `ToyODE` Neural ODE with RK4 solver.

```python
from MIOFlow.models import make_model
model = make_model(
    feature_dims=5,       # number of input dimensions (embedding dims)
    layers=[16, 32, 16],  # hidden layer sizes
    activation='CELU',    # activation function
    scales=None,          # SDE noise scales (None for pure ODE)
    use_cuda=True
)
```

Architecture: input = `feature_dims + 1 (time) + 2 (augmentation)` -> hidden layers -> output = `feature_dims`.

### `training_regimen(...)`

Three-phase training: local (t_i -> t_{i+1}), global (full trajectory ODE), post-local.

```python
from MIOFlow.train import training_regimen
local_losses, batch_losses, globe_losses = training_regimen(
    n_local_epochs=40, n_epochs=40, n_post_local_epochs=0,
    exp_dir=exp_dir,
    model=model, df=df, groups=groups, optimizer=optimizer,
    criterion=criterion, use_cuda=use_cuda,
    hold_one_out=False, hold_out=3,
    use_density_loss=True, lambda_density=20,
    autoencoder=None, use_emb=False, use_gae=False,
    sample_size=(100,),
    logger=None,
    reverse_schema=False, reverse_n=2,
    plot_every=20,
    n_points=1000, n_trajectories=100, n_bins=100,
)
```

### `generate_points(model, df, n_points, sample_time, ...)`

Samples `n_points` from the earliest timepoint in `df` and integrates the ODE forward.

```python
from MIOFlow.eval import generate_points
# sample_time=None -> uses group values [2, 4, 6]
# Returns shape: (len(sample_time), n_points, n_dims)
generated = generate_points(
    model, df, n_points=1000,
    use_cuda=True, samples_key='samples',
    sample_time=[2, 4, 6],       # or np.linspace(2, 6, 100) for smooth
    autoencoder=None, recon=False
)
```

### `config_criterion(criterion_name)`

```python
from MIOFlow.utils import config_criterion
criterion = config_criterion('ot')   # OT_loss (EMD-based)
criterion = config_criterion('mmd')  # MMD_loss
```

### `set_seeds(seed)`

Sets `torch.manual_seed`, `random.seed`, `np.random.seed`.

### EMD / MMD Evaluation

```python
import ot as pot
from MIOFlow.losses import MMD_loss

# EMD
a = pot.unif(n); b = pot.unif(n)
M = pot.dist(xs, xt, metric='euclidean')
emd = pot.emd2(a, b, torch.tensor(M))

# MMD
mmd_fn = MMD_loss()
mmd = mmd_fn.forward(torch.tensor(xs), torch.tensor(xt))
```

## Data Format

MIOFlow expects a `pd.DataFrame` with columns `d1, d2, ..., dN, samples`:
- `d1..dN`: float embedding coordinates
- `samples`: **integer** timepoint labels (e.g., 2, 4, 6)

```python
df = pd.DataFrame(embedding, columns=[f'd{i}' for i in range(1, n_dims + 1)])
df['samples'] = np.array(timepoints).astype(np.int32)
```

## tqdm Compatibility

MIOFlow's `train.py` imports `from tqdm.notebook import tqdm` which fails in scripts.
Monkey-patch before importing MIOFlow:

```python
import tqdm as _tqdm
import tqdm.notebook
tqdm.notebook.tqdm = _tqdm.tqdm
```

## Experiment Configurations

| Config | `obsm_key` | `n_dims` | `layers` | `activation` | `lambda_density` | `n_local` | `n_epochs` | `sample_size` |
|--------|-----------|--------|---------|------------|----------------|---------|----------|-------------|
| PCA | `X_pca` | 30 | [64,128,64] | CELU | 20 | 40 | 40 | 100 |
| DM | `DM_EigenVectors` | 10 | [16,32,16] | CELU | 20 | 40 | 40 | 100 |

## End-to-End CLI Commands

```bash
cd /rds/user/wz369/hpc-work/PINN_dynamics
conda activate mioflow

# Train PCA config
python scripts/MIOFlow/01_train.py \
    --data_path data/klein/klein_addpop.h5ad \
    --obsm_key X_pca --n_dims 30 \
    --config pca --exp_name klein_addpop_pca30 --seed 10

# Train DM config
python scripts/MIOFlow/01_train.py \
    --data_path data/klein/klein_addpop.h5ad \
    --obsm_key DM_EigenVectors --n_dims 10 \
    --config dm --exp_name klein_addpop_dm10 --seed 10

# Evaluate PCA
python scripts/MIOFlow/02_evaluate.py \
    --data_path data/klein/klein_addpop.h5ad \
    --model_dir results/MIOFlow/klein_addpop_pca30 \
    --obsm_key X_pca --n_dims 30

# Evaluate DM
python scripts/MIOFlow/02_evaluate.py \
    --data_path data/klein/klein_addpop.h5ad \
    --model_dir results/MIOFlow/klein_addpop_dm10 \
    --obsm_key DM_EigenVectors --n_dims 10
```

## Checkpoint Format

```python
ckpt = {
    "model_state_dict": model.state_dict(),
    "autoencoder_state_dict": None,
    "options": opts   # dict of all hyperparameters
}
torch.save(ckpt, f"{exp_dir}/model_checkpoints.pt")
```

## Key Implementation Notes

- `generate_points` samples from the **earliest** timepoint in the DataFrame
- `sample_time` can be group values `[2, 4, 6]` or a dense grid `np.linspace(2, 6, 100)`
- Training phases: local loss (each consecutive pair) -> global loss (full trajectory) -> post-local
- The `criterion='ot'` uses `OT_loss` which computes EMD via the POT library internally
- `lambda_density=20` adds a density matching regularization term
- Test set (Well == 2) has 0 cells at tp=2.0, so simulate from training cells at tp=2.0
