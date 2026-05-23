#!/usr/bin/env bash
#SBATCH -p ampere
#SBATCH -A GOTTGENS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --job-name=cb_tier2
#SBATCH --array=0-5                   # 3 methods × 2 embeddings = 6 tasks
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --output=logs/cordblood/slurm/tier2_3seed_%A_%a.out
#SBATCH --error=logs/cordblood/slurm/tier2_3seed_%A_%a.err
#
# Cord blood Tier-2 SLURM array, 3 seeds per task. Same shape as tier-1.
#
# Methods: mioflow, tigon, trajectorynet.
# Submit AFTER tier-1 acceptance gates pass (see docs/cordblood/task_plan.md Phase G).

set -uo pipefail

PROJ=/rds/user/wz369/hpc-work/PINN_dynamics
cd "$PROJ"

mkdir -p logs/cordblood/slurm

MANIFEST=scripts/cordblood/manifest_tier2_grouped.tsv
TASK=$SLURM_ARRAY_TASK_ID

LINE=$(sed -n "$((TASK + 2))p" "$MANIFEST")
if [[ -z "$LINE" ]]; then
    echo "Empty manifest line for task $TASK"
    exit 2
fi
read -r TASK_ID METHOD EMBEDDING OUTDIR_BASE <<< "$LINE"
[[ "$TASK_ID" == "$TASK" ]] || { echo "manifest off-by-one: row task_id=$TASK_ID, slurm task=$TASK"; exit 2; }

LOCAL_DATA="$TMPDIR/cordblood_addpop.h5ad"
cp "$PROJ/data/cordblood_addpop.h5ad" "$LOCAL_DATA"

SEEDS=(42 7 1234)
SUMMARY="logs/cordblood/$METHOD/${EMBEDDING}.summary"
mkdir -p "$(dirname "$SUMMARY")"
echo "[tier2 task $TASK] method=$METHOD emb=$EMBEDDING  $(date)  host=$(hostname)" | tee -a "$SUMMARY"

for SEED in "${SEEDS[@]}"; do
    OUTDIR="${OUTDIR_BASE}_s${SEED}"
    mkdir -p "$OUTDIR"
    LOG="logs/cordblood/$METHOD/${EMBEDDING}_s${SEED}.log"

    SLEEP_S=$((RANDOM % 30))
    echo "  [seed=$SEED]  sleep ${SLEEP_S}s, then train+eval. outdir=$OUTDIR  log=$LOG" | tee -a "$SUMMARY"
    sleep "$SLEEP_S"

    {
        echo "==================================================================="
        echo "[tier2 task $TASK / seed $SEED] $(date)"
        echo "==================================================================="
    } >> "$LOG"

    if scripts/cordblood/dispatch_task.sh "$METHOD" "$EMBEDDING" "$SEED" "$OUTDIR" "$LOCAL_DATA" >>"$LOG" 2>&1; then
        echo "    [seed=$SEED] PASS  ($(date))" | tee -a "$SUMMARY"
    else
        rc=$?
        echo "    [seed=$SEED] FAIL rc=$rc  ($(date))  — see $LOG" | tee -a "$SUMMARY"
    fi
done

echo "[tier2 task $TASK] DONE  $(date)" | tee -a "$SUMMARY"
