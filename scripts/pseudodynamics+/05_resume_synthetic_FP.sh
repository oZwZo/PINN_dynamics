#!/usr/bin/env bash
# 05_resume_synthetic_FP.sh — resume training for synthetic-FP ablation arms
#
# Finds the best existing checkpoint per seed and resumes from it.
# If no checkpoint exists for a seed, starts from scratch.
#
# Usage:
#   bash scripts/pseudodynamics+/05_resume_synthetic_FP.sh [GPU_ID]
#
set -uo pipefail

GPU=${1:-0}
PROJECT_DIR="/rds/user/wz369/hpc-work/pseudodynamics_plus"

cd "$PROJECT_DIR"

PYTHON="singularity exec --nv \
    --bind /rds/user/wz369/hpc-work/pseudodynamics_plus:/work \
    --bind /rds/user/wz369/hpc-work/pseudodynamics_plus/src/pseudodynamics:/usr/local/lib/python3.10/dist-packages/pseudodynamics \
    /home/wz369/rds/hpc-work/containers/pseudodynamics+_torch2.10.0+cu128_20260304.sif python"
MANIFEST=logs/ablation_manifest.txt
DATASET_FILTER=synthetic_FP
MAX_EPOCHS=${MAX_EPOCHS:-400}

mkdir -p logs/ablation_slurm

echo "Working directory : $PROJECT_DIR"
echo "Python            : $PYTHON"
echo "GPU               : $GPU"
echo "Manifest          : $MANIFEST"
echo "Mode              : RESUME (will find existing checkpoints)"
echo

find_best_ckpt() {
    local ckpt_dir="$1"
    if [ ! -d "$ckpt_dir" ]; then
        echo ""
        return
    fi
    # For negative val_loss (NLL), best = least negative = max value
    # Sort numerically by the val_loss value extracted from filenames
    local best=""
    local best_loss="-999999999999"
    for f in "$ckpt_dir"/*.ckpt; do
        [ -f "$f" ] || continue
        local loss
        loss=$(echo "$f" | grep -oP 'val_loss=\K-?[\d.]+(?=\.ckpt)')
        if [ -n "$loss" ]; then
            if python3 -c "exit(0 if $loss > $best_loss else 1)" 2>/dev/null; then
                best_loss="$loss"
                best="$f"
            fi
        fi
    done
    echo "$best"
}

TOTAL_START=$(date +%s)

while IFS=$'\t' read -r TASK_ID CFG DATASET ARM; do
    [[ "$DATASET" != "$DATASET_FILTER" ]] && continue

    echo "════════════════════════════════════════════════════════════════════════"
    echo "[ARM]  task=${TASK_ID}  dataset=${DATASET}  arm=${ARM}"
    echo "[CFG]  ${CFG}"
    echo "[TIME] $(date '+%F %T')"
    echo "════════════════════════════════════════════════════════════════════════"

    ARM_START=$(date +%s)

    for S in 0 1 2; do
        LOG_NAME="${DATASET}_ablation/${ARM}/seed_${S}"
        SEED_DIR="logs/${LOG_NAME}/pde_params_tsense/lightning_logs"

        # Find the latest version directory
        LATEST_VER=""
        if [ -d "$SEED_DIR" ]; then
            LATEST_VER=$(ls -d "$SEED_DIR"/version_* 2>/dev/null | sort -V | tail -1)
        fi

        RESUME_ARG=""
        if [ -n "$LATEST_VER" ]; then
            CKPT=$(find_best_ckpt "$LATEST_VER/checkpoints")
            if [ -n "$CKPT" ]; then
                RESUME_ARG="--resume_ckpt $CKPT"
                echo "[SEED] seed=${S} RESUMING from $(basename "$CKPT")"
            else
                echo "[SEED] seed=${S} no checkpoint found, starting fresh"
            fi
        else
            echo "[SEED] seed=${S} no version dir, starting fresh"
        fi

        $PYTHON main_train.py \
            --config "${CFG}" \
            -G "${GPU}" \
            --seed "${S}" \
            --log_name "${LOG_NAME}" \
            --max_epochs "${MAX_EPOCHS}" \
            --progress_bar False \
            ${RESUME_ARG} \
            > "logs/ablation_slurm/synthetic_FP_${ARM}_seed${S}.out" 2>&1 &
        echo "[SEED] launched seed=${S}  pid=$!"
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
