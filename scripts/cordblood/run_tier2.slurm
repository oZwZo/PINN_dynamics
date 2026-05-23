#!/usr/bin/env bash
#SBATCH --job-name=cordblood_tier2
#SBATCH --array=0-17                  # 3 methods × 2 embeddings × 3 seeds
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00               # max across tier-2 (TrajectoryNet PC-30 ~24h)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/cordblood/slurm/tier2_%A_%a.out
#SBATCH --error=logs/cordblood/slurm/tier2_%A_%a.err
#
# Cord blood Tier-2 array. Submitted ONLY after tier-1 acceptance gates pass.
#
# Usage:
#   sbatch scripts/cordblood/run_tier2.slurm

set -euo pipefail

PROJ=/rds/user/wz369/hpc-work/PINN_dynamics
cd "$PROJ"

mkdir -p logs/cordblood/slurm

MANIFEST=scripts/cordblood/manifest_tier2.tsv
TASK=$SLURM_ARRAY_TASK_ID

LINE=$(sed -n "$((TASK + 2))p" "$MANIFEST")
if [[ -z "$LINE" ]]; then
    echo "Empty manifest line for task $TASK — array size mismatch?"
    exit 2
fi
read -r TASK_ID METHOD EMBEDDING SEED OUTDIR <<< "$LINE"
[[ "$TASK_ID" == "$TASK" ]] || { echo "manifest off-by-one: row task_id=$TASK_ID, slurm task=$TASK"; exit 2; }

mkdir -p "$OUTDIR" "logs/cordblood/$METHOD"
LOG="logs/cordblood/$METHOD/${EMBEDDING}_s${SEED}.log"
{
    echo "==================================================================="
    echo "[tier2 task $TASK] method=$METHOD emb=$EMBEDDING seed=$SEED"
    echo "  outdir=$OUTDIR  host=$(hostname)  $(date)"
    echo "==================================================================="
} | tee -a "$LOG"

LOCAL_DATA="$TMPDIR/cordblood_addpop.h5ad"
cp "$PROJ/data/cordblood_addpop.h5ad" "$LOCAL_DATA"

scripts/cordblood/dispatch_task.sh "$METHOD" "$EMBEDDING" "$SEED" "$OUTDIR" "$LOCAL_DATA" 2>&1 | tee -a "$LOG"
echo "[tier2 task $TASK] EXIT $?" | tee -a "$LOG"
