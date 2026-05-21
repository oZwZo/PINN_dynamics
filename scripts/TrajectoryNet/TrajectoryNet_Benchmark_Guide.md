# TrajectoryNet Benchmark — Klein Dataset

## Overview

Benchmarking TrajectoryNet (Continuous Normalizing Flow) on the Klein hematopoiesis dataset (`klein_addpop.h5ad`). Two experiments: **PCA 30D** and **DM 10D**.

| Setting | Value |
|---|---|
| Training data | Well 0, 1 (~72k cells) |
| Test data | Well 2 (held out) |
| Growth model | None (base CNF only) |
| Training iterations | 10,000 |
| Evaluation metrics | W2 distance, Fate bias (Pearson r) |

---

## Prerequisites

```bash
PYTHON=/rds/user/wz369/hpc-work/LIBS/mamba/envs/mioflow/bin/python
cd /rds/user/wz369/hpc-work/PINN_dynamics
```

> **Note:** All scripts live under `scripts/TrajectoryNet/`. Data is expected at `data/klein/klein_addpop.h5ad`.

---

## Step 1: Prepare Data

Converts h5ad to TrajectoryNet NPZ format. Standardizes embeddings and maps timepoints `[2, 4, 6]` to consecutive integers `[0, 1, 2]`.

### PCA 30D

```bash
$PYTHON scripts/TrajectoryNet/01_prepare_data.py \
    --data_path data/klein/klein_addpop.h5ad \
    --output_dir logs/TrajectoryNet --run_name pca30 \
    --obsm_key X_pca --n_dims 30
```

### DM 10D

```bash
$PYTHON scripts/TrajectoryNet/01_prepare_data.py \
    --data_path data/klein/klein_addpop.h5ad \
    --output_dir logs/TrajectoryNet --run_name dm10 \
    --obsm_key DM_EigenVectors --n_dims 10
```

### Verify

```bash
$PYTHON -c "
import numpy as np
for name in ['pca30', 'dm10']:
    d = np.load(f'logs/TrajectoryNet/{name}/klein_train.npz', allow_pickle=True)
    print(f'{name}: keys={list(d.keys())}, labels={np.unique(d[\"sample_labels\"])}')
    for k in d:
        if k != 'sample_labels':
            print(f'  {k} shape: {d[k].shape}')
"
```

### Checklist

- [ ] `klein_train.npz` exists for both pca30 and dm10
- [ ] `sample_labels` are `[0, 1, 2]` (consecutive integers)
- [ ] pca30 embedding shape is `(~72k, 30)`, dm10 is `(~72k, 10)`
- [ ] `scaler.npz`, `test_cells.npz`, `train_meta.npz`, `config.json` all created

### Output Files

| File | Description |
|---|---|
| `klein_train.npz` | NPZ with `{obsm_key: embedding, sample_labels: int_labels}` |
| `scaler.npz` | Standardization mean and std from train data |
| `test_cells.npz` | Test embeddings, timepoints, cell types |
| `train_meta.npz` | Per-timepoint train embeddings and cell types (for KNN) |
| `config.json` | Experiment configuration |

---

## Step 2: Train

Thin wrapper that calls `python -m TrajectoryNet.main` via subprocess.

### PCA 30D (128-128-128 hidden)

```bash
$PYTHON scripts/TrajectoryNet/02_train.py \
    --data_path logs/TrajectoryNet/pca30/klein_train.npz \
    --embedding_name X_pca --max_dim 30 \
    --save_dir logs/TrajectoryNet/pca30/model \
    --dims 128-128-128 --niters 10000 --gpu 0
```

### DM 10D (64-64-64 hidden)

```bash
$PYTHON scripts/TrajectoryNet/02_train.py \
    --data_path logs/TrajectoryNet/dm10/klein_train.npz \
    --embedding_name DM_EigenVectors --max_dim 10 \
    --save_dir logs/TrajectoryNet/dm10/model \
    --dims 64-64-64 --niters 10000 --gpu 0
```

### Smoke Test (optional, 100 iters)

```bash
$PYTHON scripts/TrajectoryNet/02_train.py \
    --data_path logs/TrajectoryNet/pca30/klein_train.npz \
    --embedding_name X_pca --max_dim 30 \
    --save_dir logs/TrajectoryNet/pca30/model_test \
    --dims 128-128-128 --niters 100 --gpu 0
```

### Verify

```bash
ls logs/TrajectoryNet/pca30/model/checkpt.pth
ls logs/TrajectoryNet/pca30/model/train_args.json
ls logs/TrajectoryNet/dm10/model/checkpt.pth
ls logs/TrajectoryNet/dm10/model/train_args.json
```

### Checklist

- [ ] `checkpt.pth` exists in both model directories
- [ ] `train_args.json` exists in both model directories
- [ ] Training log shows decreasing loss

### Training Parameters

| Parameter | PCA 30D | DM 10D |
|---|---|---|
| Hidden dims | `128-128-128` | `64-64-64` |
| Iterations | 10,000 | 10,000 |
| Learning rate | 1e-3 | 1e-3 |
| Batch size | 1,000 | 1,000 |
| Training noise | 0.1 | 0.1 |
| Whiten | No (pre-standardized) | No (pre-standardized) |
| Growth model | No | No |

---

## Step 3: Evaluate

Loads trained model, forward-integrates from source timepoints, computes W2 distances and fate bias.

### PCA 30D

```bash
$PYTHON scripts/TrajectoryNet/03_evaluate.py \
    --data_dir logs/TrajectoryNet/pca30 \
    --model_dir logs/TrajectoryNet/pca30/model --gpu 0
```

### DM 10D

```bash
$PYTHON scripts/TrajectoryNet/03_evaluate.py \
    --data_dir logs/TrajectoryNet/dm10 \
    --model_dir logs/TrajectoryNet/dm10/model --gpu 0
```

### Verify

```bash
cat logs/TrajectoryNet/pca30/eval_results.csv
cat logs/TrajectoryNet/dm10/eval_results.csv
```

### Checklist

- [ ] `eval_results.csv` has columns: `task, replicate, w2, pearson_r, pearson_p`
- [ ] Summary rows (mean, std) appear at the bottom of each CSV
- [ ] `eval_results_fate_fractions.csv` created with per-cell-type fractions
- [ ] W2 and Pearson r values are reasonable (compare against TIGON baseline)

### Evaluation Tasks

| Task | Source | Target | Description |
|---|---|---|---|
| `tp0_to_1` | Train cells at tp=0 | Test cells at tp=1 | Day 2 to Day 4 transition |
| `tp1_to_2` | Test cells at tp=1 | Test cells at tp=2 | Day 4 to Day 6 transition |

### Evaluation Parameters

| Parameter | Default |
|---|---|
| Simulation replicates | 10 |
| Cells per replicate | 5,000 |
| KNN neighbors (fate) | 15 |

---

## Output Directory Structure

```
logs/TrajectoryNet/
├── pca30/
│   ├── klein_train.npz
│   ├── scaler.npz
│   ├── test_cells.npz
│   ├── train_meta.npz
│   ├── config.json
│   ├── model/
│   │   ├── checkpt.pth
│   │   └── train_args.json
│   ├── eval_results.csv
│   └── eval_results_fate_fractions.csv
└── dm10/
    └── [same structure]
```

---

## Key Design Decisions

### Why consecutive integer labels?

TrajectoryNet computes integration points as:

```python
int_tps = (np.arange(max(timepoints) + 1) + 1.0) * time_scale
```

With raw labels `[2, 4, 6]` this creates **7** integration points (wasteful and incorrect). With mapped labels `[0, 1, 2]` it creates exactly **3** points: `[0.5, 1.0, 1.5]`.

### Why no `--whiten`?

We pre-standardize in `01_prepare_data.py` using train-only statistics. This ensures the same scaler is applied to both train and test data, which is critical for fair evaluation.

### Why no `--use_growth`?

We benchmark the base CNF model without growth rate correction to isolate the trajectory inference capability.

### Forward integration

The evaluation uses the CNF's forward pass:

```python
cnf = model.chain[0]
z_out, delta_logp = cnf(z_in, zero, integration_times=[int_tps[i], int_tps[i+1]], reverse=False)
```

`reverse=False` integrates forward in biological time when using `int_tps` ordering.

---

## Script Reference

| Script | Purpose | Key Args |
|---|---|---|
| `01_prepare_data.py` | h5ad to NPZ conversion | `--obsm_key`, `--n_dims`, `--run_name` |
| `02_train.py` | Training wrapper (subprocess) | `--dims`, `--niters`, `--gpu` |
| `03_evaluate.py` | W2 + fate bias evaluation | `--data_dir`, `--model_dir`, `--n_sims` |
