# PRESCIENT Reference

> Source: https://cgs.csail.mit.edu/prescient/documentation/
> PRESCIENT (**P**otential ene**R**gy und**E**rlying **S**ingle-**C**ell grad**I**e**N****T**s)

---

## 1. Overview

PRESCIENT models cellular differentiation as a diffusion process governed by a stochastic ODE:

```
dx = -∇ψ(x) dt + σ dW_t
```

where the potential energy `ψ` is a scalar-valued neural network (the *AutoGenerator*).
Training uses an optimal-transport loss (Sinkhorn divergence) between simulated and observed cell distributions at successive timepoints, weighted by per-cell growth scores.

**Key capabilities**
- Simulate differentiation trajectories from any initial cell state
- Generalise to unseen initial conditions
- Perturbational analysis (in-silico gene knock-in/out)

---

## 2. Installation

```bash
# Stable release
pip install prescient

# Development version
pip install git+https://github.com/gifford-lab/prescient.git
```

Python ≥ 3.7, PyTorch ≥ 1.7 required.
Optional GPU support: CUDA-compatible PyTorch build.

---

## 3. Workflow

```
h5ad / CSV data
      │
      ▼
01. prescient process_data   →  data.pt
      │
      ▼  (optional: replace data['xp'] with custom obsm)
      │
      ▼
02. prescient train_model    →  logs/.../seed_N/train.best.pt
                                               config.pt
      │
      ▼
03. prescient simulate_trajectories  →  simulated arrays
      │   (or manual Python loop: net._step)
      ▼
04. Evaluation (W2 distance, fate bias)
```

---

## 4. CLI Reference

### 4.1 `prescient process_data`

Converts a normalized expression CSV + metadata CSV into a PRESCIENT torch object (`data.pt`).

```
prescient process_data \
    -d DATA_CSV \
    -m META_CSV \
    -o OUT_DIR \
    [--tp_col TP_COL] \
    [--celltype_col CT_COL] \
    [--num_pcs 50] \
    [--num_neighbors_umap 10] \
    [--growth_path GROWTH_PT]
```

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `data_path` | `-d` | — | **Required.** Path to normalized expression CSV (cells × genes) |
| `out_dir` | `-o` | — | **Required.** Output directory for `data.pt` |
| `meta_path` | `-m` | — | **Required.** Metadata CSV; must contain timepoint and celltype columns |
| `tp_col` | — | — | Column name for timepoints |
| `celltype_col` | — | — | Column name for cell type annotations |
| `num_pcs` | — | 50 | Number of PCA components (overridden when you replace `xp`) |
| `num_neighbors_umap` | — | 10 | UMAP neighbors for visualization embedding |
| `growth_path` | — | None | Path to pre-computed growth weights `.pt` file |

**Output file naming**: `{out_dir}/PRESCIENT_{data_stem}_{weight_stem}_data.pt`
(where `data_stem` is the stem of the expression CSV and `weight_stem` is the stem of the growth file).

**Output `data.pt` structure**:
```python
data = torch.load("data.pt")
data['xp']   # list[Tensor]  — per-timepoint PCA embeddings (cells × num_pcs)
data['xu']   # list[Tensor]  — per-timepoint UMAP coords
data['y']    # list[int]     — timepoint indices (0-based integers)
data['w']    # list[ndarray] — per-timepoint growth weights (log-scale)
data['tps']  # ndarray       — per-cell timepoint index
```

**Example**:
```bash
prescient process_data \
    -d data/klein_subset_X.csv \
    -m data/klein_subset_meta.csv \
    --growth_path logs/PRESCIENT/Msig_GOBP_cellcycle_growth.pt \
    -o logs/PRESCIENT/ \
    --tp_col timepoint_tx_days \
    --celltype_col anno_man
```

---

### 4.2 `prescient train_model`

Trains the PRESCIENT neural SDE model.

```
prescient train_model \
    -i DATA_PT \
    --out_dir OUT_DIR \
    --weight_name WEIGHT_NAME \
    [--activation softplus] \
    [--k_dim 500] \
    [--layers 2] \
    [--loss euclidean] \
    [--pretrain_lr 1e-9] \
    [--pretrain_epochs 500] \
    [--train_epochs 2500] \
    [--train_lr 0.01] \
    [--train_dt 0.1] \
    [--train_sd 0.5] \
    [--train_tau 1e-6] \
    [--train_batch 0.1] \
    [--train_clip 0.25] \
    [--save 100] \
    [--seed 1] \
    [--gpu GPU_INT]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `data_path` / `-i` | — | **Required.** PRESCIENT `data.pt` from `process_data` |
| `out_dir` | — | **Required.** Root output directory |
| `weight_name` | — | **Required.** Identifier string used in output dir name |
| `activation` | `softplus` | Network activation function |
| `k_dim` | 500 | Number of hidden units per layer |
| `layers` | 2 | Number of hidden layers |
| `loss` | `euclidean` | Distance function for optimal transport (`euclidean` or `cosine`) |
| `pretrain_lr` | `1e-9` | Learning rate during pre-training phase |
| `pretrain_epochs` | 500 | Number of pre-training epochs |
| `train_epochs` | 2500 | Number of main training epochs |
| `train_lr` | `0.01` | Learning rate during training |
| `train_dt` | `0.1` | Euler-Maruyama timestep for SDE simulation |
| `train_sd` | `0.5` | Gaussian noise standard deviation σ |
| `train_tau` | `1e-6` | Entropy regularisation parameter τ (Sinkhorn) |
| `train_batch` | `0.1` | Fraction of cells per mini-batch |
| `train_clip` | `0.25` | Gradient clipping threshold |
| `save` | 100 | Save checkpoint every N epochs |
| `seed` | 1 | Random seed |
| `gpu` | None (CPU) | CUDA device index (integer) |

**Output directory structure**:
```
{out_dir}/{weight_name}-{activation}_{layers}_{k_dim}-{train_tau}/seed_{seed}/
    train.best.pt    ← best checkpoint (lowest training loss)
    train.{N}.pt     ← checkpoint every `save` epochs
    pretrain.pt      ← pre-training checkpoint
    config.pt        ← full training config dict
    train.log        ← training loss log
    done.log         ← completion marker
```

**Example**:
```bash
# DM config
prescient train_model \
    -i logs/PRESCIENT/dm_run/train_expr_data.pt \
    --out_dir logs/PRESCIENT/dm_run/ \
    --weight_name DM \
    --activation softplus --k_dim 64 --layers 4 \
    --train_tau 1e-6 --train_dt 0.1 --train_sd 0.5 \
    --pretrain_epochs 500 --train_epochs 2500 \
    --pretrain_lr 1e-9 --train_lr 0.01 \
    --seed 2 --gpu 0

# PCA config
prescient train_model \
    -i logs/PRESCIENT/pca_run/train_expr_data.pt \
    --out_dir logs/PRESCIENT/pca_run/ \
    --weight_name PCA \
    --activation softplus --k_dim 500 --layers 1 \
    --train_tau 1e-6 --seed 2 --gpu 0
```

---

### 4.3 `prescient simulate_trajectories`

Generates simulated trajectories using a trained model.

```
prescient simulate_trajectories \
    -i DATA_PT \
    --model_path MODEL_DIR \
    -o OUT_DIR \
    [--num_sims 10] \
    [--num_cells 200] \
    [--num_steps NUM] \
    [--seed 1] \
    [--epoch 002500] \
    [--gpu GPU_INT] \
    [--celltype_subset CT] \
    [--tp_subset TP]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `data_path` / `-i` | — | **Required.** PRESCIENT `data.pt` |
| `model_path` | — | **Required.** Model directory (contains `train.best.pt`, `config.pt`) |
| `out_path` / `-o` | — | **Required.** Output directory |
| `num_sims` | 10 | Number of simulation replicates |
| `num_cells` | 200 | Cells per simulation |
| `num_steps` | auto | Forward simulation steps (auto-calculated from config if omitted) |
| `seed` | 1 | Model seed |
| `epoch` | `'002500'` | Model epoch to load (string, zero-padded) |
| `gpu` | None | CUDA device index |
| `celltype_subset` | None | Filter starting cells by cell type |
| `tp_subset` | None | Filter starting cells by timepoint |

---

### 4.4 `prescient perturbation_analysis`

Runs perturbed and unperturbed simulations for functional gene analysis.

```
prescient perturbation_analysis \
    -i DATA_PT \
    --model_path MODEL_DIR \
    -o OUT_DIR \
    -p GENE1,GENE2 \
    -z 5.0 \
    [--num_sims 10] [--num_cells 200] [--num_steps NUM] \
    [--seed 1] [--epoch 1344] [--gpu GPU_INT] \
    [--celltype_subset CT] [--tp_subset TP]
```

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `perturb_genes` | `-p` | — | **Required.** Comma-separated gene names (no spaces) |
| `z_score` | `-z` | 5.0 | Perturbation magnitude in z-score units |
| Other | — | same as `simulate_trajectories` | — |

---

## 5. Python API

### 5.1 Growth weight computation

```python
import prescient.utils

growth_weights, growth_log = prescient.utils.get_growth_weights(
    xs,                 # scaled expression DataFrame (cells × genes)
    xp,                 # PCA coords array (cells × n_pcs)
    metadata,           # obs DataFrame with timepoint column
    tp_col="Time_point",
    genes=adata.var_names,
    birth_gst="path/to/birth_geneset.csv",    # gene symbol list CSV
    death_gst="path/to/death_geneset.csv",
    outfile="growth_weights.pt",
)
# growth_weights: log-scale weights, shape (n_cells,)
```

### 5.2 Loading a trained model

```python
import torch
from prescient.train.model import SimpleNamespace, AutoGenerator

config_path = "logs/.../seed_2/config.pt"
train_pt    = "logs/.../seed_2/train.best.pt"

config  = SimpleNamespace(**torch.load(config_path))
net     = AutoGenerator(config)

checkpoint = torch.load(train_pt, map_location=device)
net.load_state_dict(checkpoint["model_state_dict"])
net = net.to(device)
net.eval()
```

**config attributes** (from saved `config.pt`):

| Attribute | Description |
|-----------|-------------|
| `x_dim` | Input/output dimensionality |
| `k_dim` | Hidden units per layer |
| `layers` | Number of hidden layers |
| `activation` | Activation function name |
| `train_dt` | Simulation timestep |
| `train_sd` | Gaussian noise std dev |
| `train_tau` | Sinkhorn entropy parameter |
| `train_t` | List of timepoint indices (e.g. `[1, 2]`) |
| `start_t` | Start timepoint index (usually `0`) |
| `seed` | Random seed |
| `data_path` | Path to training data |
| `out_dir` | Output directory |

### 5.3 Manual trajectory simulation

```python
import torch

# num_steps from start_t to last training timepoint
num_steps = int((config.train_t[-1] - config.start_t) / config.train_dt)

x_i = torch.tensor(initial_cells, dtype=torch.float32).to(device)

with torch.no_grad():
    for _ in range(num_steps):
        z   = torch.randn(x_i.shape[0], x_i.shape[1], device=device) * config.train_sd
        x_i = net._step(x_i, dt=config.train_dt, z=z)

final_cells = x_i.detach().cpu().numpy()
```

### 5.4 Wasserstein-2 distance

```python
from torchcfm.optimal_transport import wasserstein

w2 = wasserstein(x_simulated.cpu(), x_true.cpu(), power=2, method="exact")
```

---

## 6. Data Format Requirements

### Expression CSV (`-d`)
- Shape: cells × genes
- Values: **normalized** expression (e.g. `scanpy.pp.normalize_total` + `log1p`, then `StandardScaler`)
- Index: cell barcodes
- Header: gene names

### Metadata CSV (`-m`)
- Index: cell barcodes (must match expression CSV)
- Required columns: timepoint column, cell type column

### Growth weights `.pt`
- Saved via `prescient.utils.get_growth_weights(..., outfile=...)`
- Contains log-scale proliferation weights per cell
- Uses MSigDB gene sets for birth (cell cycle) and death (apoptosis) gene modules

---

## 7. Experiment Configs for This Project

| Config | `obsm_key` | `n_dims` | `k_dim` | `layers` | `activation` | `train_tau` | `train_dt` | `train_sd` |
|--------|-----------|--------|-------|--------|------------|-----------|----------|----------|
| **DM** | `DM_EigenVectors` | 10 | 64 | 4 | `softplus` | `1e-6` | `0.1` | `0.5` |
| **PCA** | `X_pca` | 30 | 500 | 1 | `softplus` | `1e-6` | `0.1` | `0.5` |

Both configs: `pretrain_epochs=500`, `train_epochs=2500`, `pretrain_lr=1e-9`, `train_lr=0.01`

### Train/test split

- **Train**: `adata.obs['Well'] != 2`
- **Test**:  `adata.obs['Well'] == 2`
- Timepoint column: `timepoint_tx_days`
- Cell type column: `anno_man`

---

## 8. End-to-End Workflow Scripts

### Step 1 — Prepare data

```bash
# PCA
python scripts/prescient/01_prepare_data.py \
    --data_path  data/klein_subset.h5ad \
    --growth_path Explore_Notebook/8.Benchmark_v/Msig_GOBP_cellcycle_growth.pt \
    --output_dir logs/PRESCIENT/ \
    --run_name   pca_run \
    --obsm_key   X_pca \
    --n_dims     30

# DM
python scripts/prescient/01_prepare_data.py \
    --data_path  data/klein_subset.h5ad \
    --growth_path Explore_Notebook/8.Benchmark_v/Msig_GOBP_cellcycle_growth.pt \
    --output_dir logs/PRESCIENT/ \
    --run_name   dm_run \
    --obsm_key   DM_EigenVectors \
    --n_dims     10
```

Outputs in `logs/PRESCIENT/{run_name}/`:
- `train_expr.csv` — training expression matrix
- `train_meta.csv` — training metadata
- `*data.pt` — PRESCIENT torch object with replaced `xp`
- `test_cells.npy`, `test_tps.npy`, `test_celltypes.npy` — held-out test set

### Step 2 — Train

```bash
python scripts/prescient/02_train.py \
    --data_path logs/PRESCIENT/pca_run/train_expr_data.pt \
    --out_dir   logs/PRESCIENT/pca_run/ \
    --config    pca \
    --seed      2 \
    --gpu       0

python scripts/prescient/02_train.py \
    --data_path logs/PRESCIENT/dm_run/train_expr_data.pt \
    --out_dir   logs/PRESCIENT/dm_run/ \
    --config    dm \
    --seed      2 \
    --gpu       0
```

### Step 3 — Evaluate (W2 + population fate bias)

```bash
python scripts/prescient/03_evaluate.py \
    --data_path data/klein_subset.h5ad \
    --model_dir logs/PRESCIENT/pca_run/PCA-softplus_1_500-1e-06/seed_2 \
    --obsm_key  X_pca --n_dims 30 \
    --output    logs/PRESCIENT/pca_run/eval_results.csv

python scripts/prescient/03_evaluate.py \
    --data_path data/klein_subset.h5ad \
    --model_dir logs/PRESCIENT/dm_run/DM-softplus_4_64-1e-06/seed_2 \
    --obsm_key  DM_EigenVectors --n_dims 10 \
    --output    logs/PRESCIENT/dm_run/eval_results.csv
```

### Step 3b — Evaluate (per-cell fate accuracy, TIGON-style)

Script: `scripts/prescient/eval_fate_acc.py`

Uses `F_obs.csv` (clone-traced ground-truth fate proportions per cell) to evaluate
how accurately PRESCIENT predicts the dominant fate of each starting cell.

```bash
python scripts/prescient/eval_fate_acc.py \
    --data_path  data/klein_subset.h5ad \
    --model_dir  logs/PRESCIENT/pca_run/PCA-softplus_1_500-1e-06/seed_2 \
    --F_obs_path data/klein/F_obs.csv \
    --obsm_key   X_pca --n_dims 30 \
    --tp_start   2 \
    --n_sims     100 \
    --output     logs/PRESCIENT/pca_run/fate_acc_results.csv

python scripts/prescient/eval_fate_acc.py \
    --data_path  data/klein_subset.h5ad \
    --model_dir  logs/PRESCIENT/dm_run/DM-softplus_4_64-1e-06/seed_2 \
    --F_obs_path data/klein/F_obs.csv \
    --obsm_key   DM_EigenVectors --n_dims 10 \
    --tp_start   2 \
    --n_sims     100 \
    --output     logs/PRESCIENT/dm_run/fate_acc_results.csv
```

**Output files** (per `--output` path):
- `fate_acc_results.csv` — accuracy, pearson_r, pearson_p, metadata
- `fate_acc_results_celltype_fractions.csv` — F_obs vs F_hat mean per cell type
- `fate_acc_results_F_hat.csv` — full per-cell predicted fate matrix

**Python API**:
```python
from scripts.prescient.eval_fate_acc import evaluate_fate_accuracy
import pandas as pd

F_obs = pd.read_csv("data/klein/F_obs.csv", index_col=0)
F_obs.index = F_obs.index.astype(str)

result = evaluate_fate_accuracy(
    adata        = adata,
    net          = net,      # loaded PRESCIENT AutoGenerator
    config       = config,   # PRESCIENT SimpleNamespace
    F_obs        = F_obs,
    obsm_key     = "X_pca",
    n_dims       = 30,
    celltype_col = "anno_man",
    tp_col       = "timepoint_tx_days",
    tp_start     = 2,        # first timepoint value (day 2)
    train_mask   = adata.obs["Well"] != 2,
    n_sims       = 100,
    k            = 20,
    device       = "cuda:0",
)
print(f"Accuracy: {result['accuracy']:.4f}")
print(f"Pearson r: {result['pearson_r']:.4f}")
# result['F_hat']         — DataFrame (cells × cell_types)
# result['F_obs_aligned'] — DataFrame (cells × cell_types, ground truth)
```

---

## 9. Troubleshooting

| Issue | Solution |
|-------|----------|
| `*data.pt` not found after `process_data` | Check PRESCIENT output dir; file is named `PRESCIENT_{data_stem}_{weight_stem}_data.pt` |
| `KeyError: 'DM_EigenVectors'` | Verify obsm key with `adata.obsm.keys()` |
| CUDA OOM during training | Reduce `--train_batch` (e.g. `0.05`) or use CPU |
| `torchcfm` not found | `pip install torchcfm`; fallback sliced W2 used in evaluate script |
| `train.best.pt` missing | Training may not have completed; check `train.log` and `done.log` |
| Growth weights shape mismatch | Recompute with `prescient.utils.get_growth_weights` using **train cells only** |

---

## 10. Citation

> Yeo GHT, Saksena SD, Gifford DK.
> "Generative modeling of single-cell time series with PRESCIENT enables prediction of cell trajectories with interventions."
> *Nature Communications*, 2021.
> https://github.com/gifford-lab/prescient
