#!/bin/bash
#SBATCH --job-name=pdp_abl_klein_nonOT
#SBATCH --account=GOTTGENS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-15
#SBATCH --output=/rds/user/wz369/hpc-work/pseudodynamics_plus/logs/ablation_slurm/klein_nonOT_%A_%a.out
#SBATCH --error=/rds/user/wz369/hpc-work/pseudodynamics_plus/logs/ablation_slurm/klein_nonOT_%A_%a.err

set -u
cd /rds/user/wz369/hpc-work/pseudodynamics_plus

mkdir -p logs/ablation_slurm

# Mamba env hits 'Illegal instruction' (SIGILL) on some CSD3 ampere nodes,
# so we run inside the project's pre-built singularity container instead.
# PYTHON=/rds/user/wz369/hpc-work/LIBS/mamba/envs/PINN_env/bin/python
# PYTHON=/local/scratch/wz369/PINN_env/bin/python
PYTHON="singularity exec --nv \
    --bind /rds/user/wz369/hpc-work/pseudodynamics_plus:/work \
    --bind /rds/user/wz369/hpc-work/pseudodynamics_plus/src/pseudodynamics:/usr/local/lib/python3.10/dist-packages/pseudodynamics \
    /rds/user/wz369/hpc-work/containers/pseudodynamics+_torch2.10.0+cu128_20260304.sif python"
MANIFEST=logs/ablation_manifest.txt
DATASET_FILTER=klein_nonOT
MAX_EPOCHS=${MAX_EPOCHS:-400}

# ---------------------------------------------------------------------------
# Pick the N-th klein_nonOT-only manifest line (N = SLURM_ARRAY_TASK_ID, 0-indexed)
# ---------------------------------------------------------------------------
LINE=$(awk -v id="${SLURM_ARRAY_TASK_ID}" -v ds="${DATASET_FILTER}" '
    $3 == ds { if (n++ == id) { print; exit } }
' "${MANIFEST}")

if [[ -z "${LINE}" ]]; then
    echo "[ERROR] No ${DATASET_FILTER} manifest entry for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

CFG=$(      echo "${LINE}" | cut -f2)
DATASET=$(  echo "${LINE}" | cut -f3)
ARM=$(      echo "${LINE}" | cut -f4)

echo "[JOB]  job=${SLURM_ARRAY_JOB_ID}  task=${SLURM_ARRAY_TASK_ID}  dataset=${DATASET}  arm=${ARM}  node=$(hostname)  start=$(date '+%F %T')"
echo "[CFG]  ${CFG}"
nvidia-smi -L || true

START_SECONDS=${SECONDS}

# ---------------------------------------------------------------------------
# Launch 3 seeds in parallel on the same GPU
# ---------------------------------------------------------------------------
for S in 0 1 2; do
    $PYTHON main_train.py \
        --config "${CFG}" \
        -G 0 \
        --seed "${S}" \
        --log_name "${DATASET}_ablation/${ARM}/seed_${S}" \
        --max_epochs "${MAX_EPOCHS}" \
        --progress_bar False \
        > "logs/ablation_slurm/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${DATASET}_seed${S}.out" 2>&1 &
    echo "[SEED] launched seed=${S}  pid=$!"
    # Stagger to avoid 'CUDA-capable device busy' race when 3 processes
    # call cudaSetDevice simultaneously on the same GPU.
    sleep 15
done

# Wait for all three seeds to finish
wait

ELAPSED=$(( SECONDS - START_SECONDS ))
echo "[DONE]  job=${SLURM_ARRAY_JOB_ID}  task=${SLURM_ARRAY_TASK_ID}  dataset=${DATASET}  arm=${ARM}  elapsed=${ELAPSED}s  end=$(date '+%F %T')"
