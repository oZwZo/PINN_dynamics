#!/bin/bash
#SBATCH -p ampere
#SBATCH -A GOTTGENS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-q-[15,64,68-75,81-82,86-90]
#SBATCH --array=0-2
#SBATCH --time=24:00:00
#SBATCH --job-name scdiffeq_cb_pca30
#SBATCH --output /rds/user/wz369/hpc-work/PINN_dynamics/logs/cordblood/scdiffeq/scdiffeq_cb_pca30_seeds-%A_%a.log

# scDiffEq cord-blood PCA-30 — 3 seeds (42, 7, 1234) as SLURM array tasks.
# (Previously 3 seeds in parallel on one GPU; that crashed with
#  `torch.AcceleratorError: CUDA-capable device(s) is/are busy or unavailable`
#  because Lightning's accelerator.setup_device() races on a shared GPU.)

set -uo pipefail

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

PROJ=/rds/user/wz369/hpc-work/PINN_dynamics
SCDIFFEQ_DIR=/rds/user/wz369/hpc-work/scDiffEq
SCRIPT="$PROJ/scripts/scdiffeq/run_scdiffeq_cordblood.py"
EVAL_SCRIPT="$PROJ/scripts/scdiffeq/evaluate_scdiffeq_cordblood.py"
DATA="$PROJ/data/cordblood_addpop.h5ad"
F_OBS="$PROJ/data/CordBlood/F_obs_day17.csv"
CLONE_PROP="$PROJ/data/CordBlood/clone_proportions_day17.csv"
LOGDIR="$PROJ/logs/cordblood/scdiffeq"

mkdir -p "$LOGDIR"

SEEDS=(42 7 1234)
CONFIG=pca30
seed=${SEEDS[$SLURM_ARRAY_TASK_ID]}
log="$LOGDIR/scdiffeq_cb_${CONFIG}_s${seed}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
outdir="$LOGDIR/${CONFIG}_fp_vr_s${seed}"

echo "[$(date)] Job $SLURM_JOB_ID array task $SLURM_ARRAY_TASK_ID on $(hostname)  GPU=${CUDA_VISIBLE_DEVICES:-unset}  seed=$seed"
mkdir -p "$outdir"

# Small jitter so 3 array tasks don't hit the shared data-load step simultaneously.
sleep $((RANDOM % 15))

uv run --directory "$SCDIFFEQ_DIR" python "$SCRIPT" \
    --config "$CONFIG" \
    --data   "$DATA" \
    --output_dir "$LOGDIR" \
    --epochs 2500 \
    --seed   "$seed" > "$log" 2>&1

CKPT=$(find "$outdir" -name "*.ckpt" ! -name "last.ckpt" | sort | tail -1)
if [[ -z "$CKPT" ]]; then
    CKPT=$(find "$outdir" -name "last.ckpt" | head -1)
fi
if [[ -n "$CKPT" ]]; then
    uv run --directory "$SCDIFFEQ_DIR" python "$EVAL_SCRIPT" \
        --config "$CONFIG" \
        --data   "$DATA" \
        --ckpt_path "$CKPT" \
        --f_obs  "$F_OBS" \
        --clone_proportions "$CLONE_PROP" \
        --output_dir "$outdir/eval" \
        --seed "$seed" >> "$log" 2>&1
else
    echo "[WARN] No checkpoint found in $outdir — skipping eval" >&2
fi
echo "[$(date)] seed=$seed task done."
