#!/usr/bin/env bash
# 05_train_ablation_synthetic_FP.sh — train the 4 synthetic-FP ablation arms
#
# Each arm trains 3 seeds in parallel on a single GPU, then waits for all
# 3 to finish before moving on to the next arm. Run interactively in a
# bash shell (not via SLURM).
#
# Usage:
#   bash scripts/pseudodynamics+/05_train_ablation_synthetic_FP.sh [GPU_ID]
#
# GPU_ID defaults to 0.
#
set -uo pipefail

GPU=${1:-0}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_DIR"

# Mamba env hits 'Illegal instruction' (SIGILL) on some CSD3 ampere nodes,
# so we run inside the project's pre-built singularity container instead.
# PYTHON=/rds/user/wz369/hpc-work/LIBS/mamba/envs/PINN_env/bin/python
# PYTHON=/local/scratch/wz369/PINN_env/bin/python
PYTHON="singularity exec --nv \
    --bind /rds/user/wz369/hpc-work/pseudodynamics_plus:/work \
    --bind /rds/user/wz369/hpc-work/pseudodynamics_plus/src/pseudodynamics:/usr/local/lib/python3.10/dist-packages/pseudodynamics \
    /rds/user/wz369/hpc-work/containers/pseudodynamics+_torch2.10.0+cu128_20260304.sif python"
MANIFEST=logs/ablation_manifest.txt
DATASET_FILTER=synthetic_FP
MAX_EPOCHS=${MAX_EPOCHS:-400}

mkdir -p logs/ablation_slurm

echo "Working directory : $PROJECT_DIR"
echo "Python            : $PYTHON"
echo "GPU               : $GPU"
echo "Manifest          : $MANIFEST"
echo

TOTAL_START=$(date +%s)

# Iterate every manifest row whose dataset column == synthetic_FP
while IFS=$'\t' read -r TASK_ID CFG DATASET ARM; do
    [[ "$DATASET" != "$DATASET_FILTER" ]] && continue

    echo "════════════════════════════════════════════════════════════════════════"
    echo "[ARM]  task=${TASK_ID}  dataset=${DATASET}  arm=${ARM}"
    echo "[CFG]  ${CFG}"
    echo "[TIME] $(date '+%F %T')"
    echo "════════════════════════════════════════════════════════════════════════"

    ARM_START=$(date +%s)

    for S in 0 1 2; do
        $PYTHON main_train.py \
            --config "${CFG}" \
            -G "${GPU}" \
            --seed "${S}" \
            --log_name "${DATASET}_ablation/${ARM}/seed_${S}" \
            --max_epochs "${MAX_EPOCHS}" \
            --progress_bar False \
            > "logs/ablation_slurm/synthetic_FP_${ARM}_seed${S}.out" 2>&1 &
        echo "[SEED] launched seed=${S}  pid=$!"
        # Stagger to avoid 'CUDA-capable device busy' race
        sleep 15
    done

    wait

    ARM_END=$(date +%s)
    echo "[DONE] arm=${ARM}  elapsed=$((ARM_END - ARM_START))s  end=$(date '+%F %T')"
    echo
done < "$MANIFEST"

TOTAL_END=$(date +%s)
TOTAL_SEC=$((TOTAL_END - TOTAL_START))
echo "════════════════════════════════════════════════════════════════════════"
echo "  All synthetic_FP arms finished."
echo "  Total elapsed: $((TOTAL_SEC / 60))m $((TOTAL_SEC % 60))s"
echo "════════════════════════════════════════════════════════════════════════"
