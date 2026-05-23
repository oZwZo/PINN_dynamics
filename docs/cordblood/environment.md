# Cord Blood Benchmark — Python Environments per Method

Every method in the benchmark uses a different Python environment. This document records what each method expects, derived from the existing Klein / Weinreb scripts in `scripts/<method>/`. The cord-blood dispatcher (`scripts/cordblood/dispatch_task.sh`) must respect these choices.

## Summary table

| Method | Env | Source path | Notes |
|---|---|---|---|
| **pseudodynamics+** (legacy) | mamba env `PINN_env` | `/rds/user/wz369/hpc-work/LIBS/mamba/envs/PINN_env/bin/python` | Used by `scripts/pseudodynamics+/02_train_klein.sh` and the smoke-test SLURM. |
| **pseudodynamics+** (CFM trainer) | singularity `flow.sif` | `singularity exec --nv /rds/user/wz369/hpc-work/containers/flow.sif python` | Used by the newer `05_train_ablation_klein_*.slurm` jobs (the OT/CFM variant). The CFM-aware trainer lives in `pseudodynamics_plus/main_train.py`. |
| **PRESCIENT** | singularity `flow.sif` | `singularity exec --nv /rds/user/wz369/hpc-work/containers/flow.sif python` | All train/eval shell + slurm scripts under `scripts/prescient/`. |
| **OT-CFM** | singularity `flow.sif` | `singularity exec --nv /rds/user/wz369/hpc-work/containers/flow.sif python` | `scripts/otcfm/train_klein.sh`, `scripts/otcfm/eval_klein.sh`. |
| **SF2M** | singularity `flow.sif` | `singularity exec --nv /rds/user/wz369/hpc-work/containers/flow.sif python` | `scripts/sf2m/train_klein*.sh`, `scripts/sf2m/eval_klein*.sh`. |
| **DeepRUOT (eval-only)** | singularity `flow.sif` | `singularity exec --nv /rds/user/wz369/hpc-work/containers/flow.sif python` | `scripts/DeepRUOT/eval_klein.sh` (relative path `../containers/flow.sif` from script). Training happens externally at `/rds/user/wz369/hpc-work/DeepRUOTv2/`. |
| **MIOFlow (+ GAE)** | mamba env `mioflow` | `/rds/user/wz369/hpc-work/LIBS/mamba/envs/mioflow/bin/python` | `scripts/MIOFlow/01_train_klein_*.sh`, `scripts/MIOFlow/03_evaluate.sh`. |
| **TIGON** | mamba env `mioflow` | `/rds/user/wz369/hpc-work/LIBS/mamba/envs/mioflow/bin/python` | `scripts/TIGON/02_train_tigon.sh`, `scripts/TIGON/03_evaluate.sh`. Reuses MIOFlow env. |
| **TrajectoryNet** | mamba env `mioflow` | `/rds/user/wz369/hpc-work/LIBS/mamba/envs/mioflow/bin/python` | `scripts/TrajectoryNet/evalute_klein.sh`. Also uses `mioflow` env per shebang in `scripts/TIGON/01_prepare_data.py`. |
| **scdiffeq (eval-only)** | uv env at `/rds/user/wz369/hpc-work/scDiffEq/` | `uv run python` (cwd = `$UV_PROJECT`) | `scripts/scdiffeq/eval_klein.sh` / `submit_klein_fate_eval.sbatch`. |

## Cluster-side environment groups

There are **three** runtime environments in active use:

### 1. `PINN_env` (mamba)
- Path: `/rds/user/wz369/hpc-work/LIBS/mamba/envs/PINN_env/bin/python`
- Used by: legacy pseudodynamics+ trainer + our cord-blood preprocessing (`scripts/cordblood/01_preprocess.py`, `00_inspect.py`).
- Verified: `scanpy 1.10.3`, `anndata 0.10.9`, `palantir 1.4.0`.
- Also where `PINN` package import works (used in `01_preprocess.py:_sample_deltax_stack` via `import PINN as pdp`).

### 2. `mioflow` (mamba)
- Path: `/rds/user/wz369/hpc-work/LIBS/mamba/envs/mioflow/bin/python`
- Used by: MIOFlow, TIGON, TrajectoryNet (training and eval).
- This env has the MIOFlow + Graph Auto-Encoder dependencies.

### 3. `flow.sif` (singularity container)
- Path: `/rds/user/wz369/hpc-work/containers/flow.sif`
- Wrapped command: `singularity exec --nv /rds/user/wz369/hpc-work/containers/flow.sif python`
- Used by: PRESCIENT, OT-CFM, SF2M, DeepRUOT, and the newer pseudodynamics+ CFM trainer.
- `--nv` enables GPU access in the container.

### 4. `scDiffEq` uv project (sub-case)
- Activated via `uv run python` with `--directory /rds/user/wz369/hpc-work/scDiffEq`.
- Used only by `scripts/scdiffeq/klein_fate_eval_scdiffeq.py`.

## Cord-blood dispatcher mapping

The dispatcher `scripts/cordblood/dispatch_task.sh` selects the right env per method. Today it uses two variables:

```bash
PYTHON_PINN=/rds/user/wz369/hpc-work/LIBS/mamba/envs/PINN_env/bin/python
PYTHON_MIOFLOW=/rds/user/wz369/hpc-work/LIBS/mamba/envs/mioflow/bin/python
```

It needs to also know about the singularity container. Recommended additions to the dispatcher:

```bash
PYTHON_FLOW="singularity exec --nv /rds/user/wz369/hpc-work/containers/flow.sif python"
PYTHON_SCDIFFEQ="uv run --directory /rds/user/wz369/hpc-work/scDiffEq python"
```

Mapping for cord-blood tasks:

| Method | Dispatcher variable to use |
|---|---|
| `pdp` (CFM trainer in `pseudodynamics_plus`) | `$PYTHON_FLOW` (singularity) — _or_ `$PYTHON_PINN` for the legacy non-CFM trainer if you skip CFM |
| `prescient` | `$PYTHON_FLOW` |
| `otcfm` | `$PYTHON_FLOW` |
| `sf2m` | `$PYTHON_FLOW` |
| `deepruot` (eval) | `$PYTHON_FLOW` |
| `mioflow` | `$PYTHON_MIOFLOW` |
| `tigon` | `$PYTHON_MIOFLOW` |
| `trajectorynet` | `$PYTHON_MIOFLOW` |
| `scdiffeq` (eval) | `$PYTHON_SCDIFFEQ` (`uv run`) |

**Current state of dispatcher (2026-05-22):** uses `PYTHON_PINN` for pdp/PRESCIENT/OT-CFM/SF2M/TIGON/TrajectoryNet/scdiffeq and `PYTHON_MIOFLOW` only for MIOFlow. This is INCORRECT for everything that needs `flow.sif`. Needs fixing — see `scripts/cordblood/dispatch_task.sh`.

## Environments NOT yet known

- `DeepRUOTv2` external training: the train invocation lives in `/rds/user/wz369/hpc-work/DeepRUOTv2/` — its own env (presumably `flow.sif` or a project-local conda). Cord-blood DeepRUOT eval pulls from those checkpoints (`/rds/user/wz369/hpc-work/DeepRUOTv2/results/cordblood_*`).

## How to verify each env is available

```bash
# PINN_env
/rds/user/wz369/hpc-work/LIBS/mamba/envs/PINN_env/bin/python -c \
    "import scanpy, anndata, palantir; print('PINN_env OK')"

# mioflow
/rds/user/wz369/hpc-work/LIBS/mamba/envs/mioflow/bin/python -c \
    "import scanpy, anndata; print('mioflow OK')"

# flow.sif (requires singularity on PATH)
singularity exec --nv /rds/user/wz369/hpc-work/containers/flow.sif python -c \
    "import torch; print('flow.sif OK', torch.__version__)"

# scDiffEq uv
cd /rds/user/wz369/hpc-work/scDiffEq && uv run python -c \
    "import scdiffeq; print('scdiffeq OK')"
```

## Provenance

Generated from a grep across `scripts/*/{*.sh,*.slurm,*.sbatch}` for `envs/`, `singularity`, `python=`, `PYTHON=`, `uv run`. See `docs/cordblood/progress.md` for the audit step.
