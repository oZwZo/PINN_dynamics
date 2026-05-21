#!/bin/bash
#SBATCH -p ampere
#SBATCH -A GOTTGENS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --job-name=pdp_c1_synds
#SBATCH --output=logs/synthetic_FP_5D_synds/slurm/%x_%A_%a.out
#SBATCH --error=logs/synthetic_FP_5D_synds/slurm/%x_%A_%a.err
#SBATCH --time=05:00:00
# --- array range set by submission: pilot = 0, full = 1-2 ---
##SBATCH --array=0-2

# C1 λ_growth sweep on the 5-D synthetic FP dataset using the Syn_DS reader
# (matches Sup Fig 2 / 2.train_model.py).
# Each task = 1 seed × 5 λ_growth values, run in parallel on 1× A100.
#
# Usage:
#   sbatch --array=0   scripts/rebuttal_C1C2/c1_synds_slurm.sh   # pilot (seed 0)
#   sbatch --array=1-2 scripts/rebuttal_C1C2/c1_synds_slurm.sh   # remaining seeds
#
# To override max_epochs (e.g. for the GPU 2-epoch dry-run gate):
#   MAX_EPOCHS=2 sbatch --time=00:30:00 --array=0 scripts/rebuttal_C1C2/c1_synds_slurm.sh

set -euo pipefail

REPO="/rds/user/wz369/hpc-work/pseudodynamics_plus"
PYTHON="/rds/user/wz369/hpc-work/LIBS/mamba/envs/PINN_env/bin/python"
DATA="${REPO}/data/synthesized_data/5Dim_ncs_synparam_Jan23"
LAMBDAS=(0 1 3 10 100)
MAX_EPOCHS="${MAX_EPOCHS:-300}"

cd "$REPO"
mkdir -p logs/synthetic_FP_5D_synds/slurm

# ─────────────────────────────────────────────────────────────────────────────
# Preflight 1: VRAM headroom (R-gpu-oom-5x-parallel)
# ─────────────────────────────────────────────────────────────────────────────
free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if (( free_mb < 40000 )); then
    echo "[preflight] GPU has only ${free_mb} MB free; need ≥40000 for 5 parallel" >&2
    exit 2
fi
echo "[preflight] GPU free = ${free_mb} MB"

# ─────────────────────────────────────────────────────────────────────────────
# Preflight 2: data integrity + time-axis cross-check
# (R-deltax-shape, R-time-axis-alignment)
# ─────────────────────────────────────────────────────────────────────────────
"$PYTHON" - <<PY
import numpy as np, torch, scanpy as sc
init = torch.load("${DATA}/syn_init_condition_0-4.ckpt", map_location='cpu', weights_only=False)
res  = torch.load("${DATA}/syn_result_0-4.ckpt",         map_location='cpu', weights_only=False)
cs, u, t, dx = init['cellstate'], res['u_simulate'], res['integrate_time'], res['delta_x']
assert cs.ndim == 2 and cs.shape[1] == 5, cs.shape
assert u.ndim == 2 and u.shape[0] == len(t), (u.shape, t.shape)
assert dx.ndim == 2 and dx.shape == cs.shape, (dx.shape, cs.shape)
a = sc.read_h5ad("${REPO}/data/synthetic_FP_5D.h5ad")
pop_t = np.asarray(a.uns['pop']['t'])
ct = t.numpy() if hasattr(t,'numpy') else np.asarray(t)
assert np.allclose(ct, pop_t, atol=1e-6), f"time-axis drift: ckpt={ct.tolist()} h5ad={pop_t.tolist()}"
print('preflight OK', cs.shape, u.shape, t.shape, dx.shape, '| time-axis matches h5ad')
PY

# ─────────────────────────────────────────────────────────────────────────────
# Launch 5 trainings in parallel with deterministic stagger
# ─────────────────────────────────────────────────────────────────────────────
SEED="${SLURM_ARRAY_TASK_ID:-0}"
pids=()
arms=()

for i in "${!LAMBDAS[@]}"; do
    sleep $((i * 15))
    LG="${LAMBDAS[i]}"
    SAVE="logs/synthetic_FP_5D_synds/lambdag_${LG}/seed_${SEED}"
    OUT="logs/synthetic_FP_5D_synds/slurm/seed${SEED}_lambdag_${LG}.out"
    mkdir -p "$SAVE"

    echo "[task $SEED] start lambdag_${LG}  →  $OUT"
    "$PYTHON" scripts/rebuttal_C1C2/c1_train_synds.py \
        --lambda_growth "$LG" \
        --seed "$SEED" \
        --save_dir "${REPO}/${SAVE}" \
        --max_epochs "$MAX_EPOCHS" \
        > "$OUT" 2>&1 &
    pids+=($!)
    arms+=("lambdag_${LG}")
done

# In-pilot VRAM probe at +60 s (R-gpu-oom-5x-parallel mitigation 3)
( sleep 60 && echo "=== nvidia-smi @60s ===" >&2 && nvidia-smi >&2 ) &

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
