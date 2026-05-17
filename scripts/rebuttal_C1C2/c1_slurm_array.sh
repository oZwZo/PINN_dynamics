#!/bin/bash
#SBATCH -p ampere
#SBATCH -A GOTTGENS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --job-name=pdp_c1_lambdag
#SBATCH --output=logs/synthetic_FP_5D_ablation/slurm/%x_%A_%a.out
#SBATCH --error=logs/synthetic_FP_5D_ablation/slurm/%x_%A_%a.err
#SBATCH --time=03:00:00
# --- array range set by submission: pilot = 0, full = 1-2 (after pilot validates) ---
##SBATCH --array=0-2

# C1 λ_growth sweep on the 5-D synthetic Fokker–Planck dataset (Sup Fig 2).
# Each task = 1 seed × 5 λ_growth values, run in parallel on 1× A100.
#
# Usage:
#   sbatch --array=0   scripts/rebuttal_C1C2/c1_slurm_array.sh   # pilot (seed 0)
#   sbatch --array=1-2 scripts/rebuttal_C1C2/c1_slurm_array.sh   # remaining seeds

set -euo pipefail

REPO_ROOT="/rds/user/wz369/hpc-work/pseudodynamics_plus"
PYTHON="/rds/user/wz369/hpc-work/LIBS/mamba/envs/PINN_env/bin/python"
DST_H5AD="${REPO_ROOT}/data/synthetic_FP_5D.h5ad"
INDEX_TSV="${REPO_ROOT}/logs/synthetic_FP_5D_ablation/array_index.tsv"

cd "$REPO_ROOT"
mkdir -p logs/synthetic_FP_5D_ablation/slurm

# ─────────────────────────────────────────────────────────────────────────────
# Preflight: defensive integrity check on the prepared h5ad.
# The file is built once by c1_make_configs.py / c1_prepare_h5ad.py — we do
# NOT (re)build it here to avoid races across 3 concurrent array tasks.
# ─────────────────────────────────────────────────────────────────────────────
if [[ ! -r "$DST_H5AD" ]]; then
    echo "[preflight] $DST_H5AD missing or unreadable." >&2
    echo "  Run first: python scripts/rebuttal_C1C2/c1_prepare_h5ad.py" >&2
    exit 1
fi

"$PYTHON" - <<'PY'
import scanpy as sc, numpy as np
a = sc.read_h5ad('data/synthetic_FP_5D.h5ad')
# obsm
assert 'Delta_DM' in a.obsm, list(a.obsm)
dx = np.asarray(a.obsm['Delta_DM'])
assert dx.shape == (a.n_obs, 5) and np.isfinite(dx).all() and dx.std() > 0
assert 'X_data' in a.obsm and a.obsm['X_data'].shape == (a.n_obs, 5)
# obs (reader hard-codes 'timepoint_tx_days')
assert 'timepoint_tx_days' in a.obs, list(a.obs.columns)
# uns['pop']: every key the reader will read at init
pop = a.uns['pop']
for k in ('t', 'mean', 'std', 'var', 'n_lib'):
    assert k in pop, f'pop missing {k!r}; have {list(pop)}'
pop_t = np.asarray(pop['t']); obs_t = np.asarray(a.obs['timepoint_tx_days'].values)
for t in pop_t:
    assert (obs_t == t).any(), f'no cell at timepoint_tx_days == {t}'
print('preflight OK  shape=%s  obsm=%s  n_t=%d  pop=%s' % (
    a.shape, list(a.obsm), len(pop_t), list(pop)))
PY

# ─────────────────────────────────────────────────────────────────────────────
# Pull this task's 5 config paths from the array_index.tsv (one row per seed)
# ─────────────────────────────────────────────────────────────────────────────
SEED="${SLURM_ARRAY_TASK_ID:-0}"
mapfile -t CFGS < <(awk -v s="$SEED" 'NR==1{next} $1==s {for(i=2;i<=NF;i++) print $i}' "$INDEX_TSV")
if [[ ${#CFGS[@]} -ne 5 ]]; then
    echo "[ERROR] expected 5 config paths for seed=$SEED in $INDEX_TSV, got ${#CFGS[@]}" >&2
    exit 1
fi

echo "[task $SEED] launching ${#CFGS[@]} parallel trainings"

# ─────────────────────────────────────────────────────────────────────────────
# Launch the 5 trainings in parallel with deterministic stagger (R3-fix).
# Capture PIDs; propagate any non-zero exit code at the end (R3-fix).
# ─────────────────────────────────────────────────────────────────────────────
pids=()
arms=()
for i in "${!CFGS[@]}"; do
    sleep $((i * 15))                                # deterministic stagger
    ARM="$(basename "$(dirname "${CFGS[i]}")")"
    # main_train.py:123 builds save_path as
    #   {repo}/logs/{log_name}/{model}_tsense
    # so --log_name must NOT include a leading 'logs/' nor a trailing
    # 'pde_params_tsense' (otherwise both get duplicated).
    LOG_NAME="synthetic_FP_5D_ablation/${ARM}/seed_${SEED}"
    OUT_FILE="logs/synthetic_FP_5D_ablation/slurm/seed${SEED}_${ARM}.out"
    mkdir -p "logs/${LOG_NAME}/pde_params_tsense"

    echo "[task $SEED] start $ARM  →  $OUT_FILE"
    "$PYTHON" main_train.py \
        --config "${CFGS[i]}" \
        --gpu_devices 0 \
        --seed "$SEED" \
        --log_name "$LOG_NAME" \
        --progress_bar False \
        --max_epochs 300 \
        > "$OUT_FILE" 2>&1 &
    pids+=($!)
    arms+=("$ARM")
done

# Wait for each PID; collect failures.
fail=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[i]}"; then
        echo "[task $SEED] FAILED arm=${arms[i]} pid=${pids[i]}" >&2
        fail=$((fail + 1))
    else
        echo "[task $SEED] OK arm=${arms[i]}"
    fi
done

echo "[task $SEED] finished with $fail failures of ${#pids[@]}"
exit "$fail"
