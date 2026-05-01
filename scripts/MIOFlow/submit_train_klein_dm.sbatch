#!/bin/bash
#! ============================================================================
#! submit_train_klein_dm.sbatch
#!
#! Single-GPU SLURM job — train MIOFlow on Klein DM_EigenVectors for 3 seeds
#! concurrently on the same GPU (model is small enough to share an A100).
#!
#! Runs:
#!   klein_dm_gaga_run1  seed 42
#!   klein_dm_gaga_run2  seed 0
#!   klein_dm_gaga_run3  seed 2
#!
#! Submit:
#!     sbatch /home/wz369/rds/hpc-work/PINN_dynamics/scripts/MIOFlow/submit_train_klein_dm.sbatch
#! Monitor:
#!     squeue -u $USER
#!     tail -f /home/wz369/rds/hpc-work/PINN_dynamics/logs/mioflow_dm_<jobid>.out
#! ============================================================================

#SBATCH --job-name=mioflow_dm
#SBATCH --account=GOTTGENS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/wz369/rds/hpc-work/PINN_dynamics/logs/mioflow_dm_%j.out
#SBATCH --error=/home/wz369/rds/hpc-work/PINN_dynamics/logs/mioflow_dm_%j.err

set -euo pipefail

#! --- paths ------------------------------------------------------------------
PROJECT=/home/wz369/rds/hpc-work/PINN_dynamics
PYTHON=/rds/user/wz369/hpc-work/LIBS/mamba/envs/mioflow/bin/python
TRAIN=$PROJECT/scripts/MIOFlow/01_train_with_gae.py
LOG_DIR=$PROJECT/logs
mkdir -p "$LOG_DIR"

#! --- env diagnostics --------------------------------------------------------
echo "============================================================"
echo " Job        : ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo " Node       : $(hostname)"
echo " GPUs       : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo " Start      : $(date -Iseconds)"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || true
echo

#! --- run --------------------------------------------------------------------
cd "$PROJECT"

EXP_NAMES=(klein_dm_gaga_run1 klein_dm_gaga_run2 klein_dm_gaga_run3)
SEEDS=(42 0 2)
PIDS=()

for i in 0 1 2; do
    EXP=${EXP_NAMES[$i]}
    SEED=${SEEDS[$i]}
    LOG=$LOG_DIR/mioflow_dm_${SLURM_JOB_ID}_${EXP}.log
    echo "Launching ${EXP} (seed=${SEED}) → ${LOG}"
    "$PYTHON" "$TRAIN" \
        --gaga_latent_dim 10 \
        --data_path data/klein_addpop.h5ad \
        --obsm_key DM_EigenVectors \
        --obs_time_key Time_point \
        --exp_name "$EXP" \
        --seed "$SEED" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

# Wait for all three; propagate failure if any exits non-zero.
RC=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "Process ${pid} failed."
        RC=1
    fi
done

echo
echo "============================================================"
echo " End        : $(date -Iseconds)"
echo " Exit code  : ${RC}"
echo "============================================================"
exit "$RC"
