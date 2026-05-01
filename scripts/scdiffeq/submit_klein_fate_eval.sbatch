#!/bin/bash
#! ============================================================================
#! submit_klein_fate_eval.sbatch
#!
#! SLURM ARRAY JOB — evaluate 16 scDiffEq checkpoints in parallel.
#!
#!   indices 0–11 : seeded sweep
#!                   3 seeds × {pca30, dm10} × {plain_ode, plain_sde}
#!   indices 12–15: fp_vr (LightningSDE-FixedPotential-
#!                  RegularizedVelocityRatio), versions 0–3, pca30 only
#!
#! Output layout (per task):
#!   {OUT}/fate_prediction_metrics/{NAME}/
#!       eval_combined.csv
#!       F_hat_{mode}.csv
#!       w2_per_clone_{mode}.csv
#!
#! Submit:
#!     sbatch /home/wz369/rds/hpc-work/PINN_dynamics/scripts/scdiffeq/submit_klein_fate_eval.sbatch
#! Monitor:
#!     squeue -u $USER
#!     tail -f /home/wz369/rds/hpc-work/PINN_dynamics/logs/sdq_fate_eval_<jobid>_<idx>.out
#! ============================================================================

#SBATCH --job-name=sdq_fate_eval
#SBATCH --account=GOTTGENS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-15
#SBATCH --output=/home/wz369/rds/hpc-work/PINN_dynamics/logs/sdq_fate_eval_%A_%a.out
#SBATCH --error=/home/wz369/rds/hpc-work/PINN_dynamics/logs/sdq_fate_eval_%A_%a.err

set -euo pipefail

#! --- paths ------------------------------------------------------------------
UV_PROJECT=/rds/user/wz369/hpc-work/scDiffEq
EVAL_SCRIPT=/home/wz369/rds/hpc-work/PINN_dynamics/scripts/scdiffeq/klein_fate_eval_scdiffeq.py
ROOT=/rds/user/wz369/hpc-work/scDiffEq/results
LOG_DIR=/home/wz369/rds/hpc-work/PINN_dynamics/logs
mkdir -p "$LOG_DIR"

#! --- eval args --------------------------------------------------------------
N=200          # SDE trajectories / cell for fate accuracy
K=15           # kNN neighbours (matches otcfm/sf2m)
SEED=0         # split reconstruction — keep constant so F_obs, x_ref, W2 cells
               # are identical across runs (apples-to-apples)
NW2=10         # SDE trajectories / cell for W2 (matches sf2m)

#! --- build the 16-checkpoint matrix -----------------------------------------
declare -a CKPT CFG OUT NAME

# Seeded sweep: 3 seeds × 2 configs × 2 variants = 12 entries
for S in 0 1 2; do
    for C in pca30 dm10; do
        for V in plain_ode plain_sde; do
            if [[ "$V" == "plain_ode" ]]; then
                CLASS=LightningODE
            else
                CLASS=LightningSDE
            fi
            CKPT+=("${ROOT}/seeds/seed_${S}/${C}/${V}/${CLASS}/version_0/checkpoints/last.ckpt")
            CFG+=("${C}")
            OUT+=("${ROOT}/seeds/seed_${S}/${C}/${V}")
            NAME+=("last")
        done
    done
done

# fp_vr (older sweep): 4 versions, pca30 only
FP_ROOT="${ROOT}/pca30/LightningSDE-FixedPotential-RegularizedVelocityRatio"
for V in 0 1 2 3; do
    CKPT+=("${FP_ROOT}/version_${V}/checkpoints/last.ckpt")
    CFG+=("pca30")
    OUT+=("${FP_ROOT}")
    NAME+=("version_${V}.last")
done

#! --- per-task parameters ----------------------------------------------------
IDX=${SLURM_ARRAY_TASK_ID}
CKPT_I="${CKPT[$IDX]}"
CFG_I="${CFG[$IDX]}"
OUT_I="${OUT[$IDX]}"
NAME_I="${NAME[$IDX]}"

if [[ ! -f "$CKPT_I" ]]; then
    echo "ERROR: checkpoint not found: $CKPT_I" >&2
    exit 1
fi

#! --- env diagnostics --------------------------------------------------------
echo "============================================================"
echo " Job        : ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) array idx ${IDX} / 15"
echo " Node       : $(hostname)"
echo " GPUs       : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo " Start      : $(date -Iseconds)"
echo " CKPT       : ${CKPT_I}"
echo " CFG        : ${CFG_I}"
echo " OUT        : ${OUT_I}/fate_prediction_metrics/${NAME_I}/"
echo " N / k      : ${N} / ${K}"
echo " NW2 / SEED : ${NW2} / ${SEED}"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || true
echo

#! --- run --------------------------------------------------------------------
cd "$UV_PROJECT"

uv run python "$EVAL_SCRIPT" \
    --config     "$CFG_I" \
    --ckpt_path  "$CKPT_I" \
    --ckpt_name  "$NAME_I" \
    --output_dir "$OUT_I" \
    --N          "$N" \
    --k          "$K" \
    --n_sims_w2  "$NW2" \
    --seed       "$SEED"

RC=$?

echo
echo "============================================================"
echo " End        : $(date -Iseconds)"
echo " Exit code  : ${RC}"
echo "============================================================"
exit "$RC"
