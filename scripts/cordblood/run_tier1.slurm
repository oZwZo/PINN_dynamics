#!/usr/bin/env bash
#SBATCH --job-name=cordblood_tier1
#SBATCH --array=0-35                  # 6 methods × 2 embeddings × 3 seeds = 36 tasks (includes deepruot/scdiffeq eval-only)
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00               # max across tier-1 methods (pdp PC-30 ~18h)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/cordblood/slurm/tier1_%A_%a.out
#SBATCH --error=logs/cordblood/slurm/tier1_%A_%a.err
#
# Cord blood Tier-1 SLURM array — runs in ONE array job covering all tier-1
# methods. Each task's failure stays isolated to its own log file.
#
# Usage:
#   sbatch scripts/cordblood/run_tier1.slurm
#
# Setup:
#   - `python scripts/cordblood/_generate_configs.py` must have been run
#     (writes pdp configs + manifest_tier1.tsv).
#   - `data/cordblood_addpop.h5ad` must exist.
#   - `--array=0-N` should match `(wc -l manifest_tier1.tsv) - 2`.

set -euo pipefail

PROJ=/rds/user/wz369/hpc-work/PINN_dynamics
cd "$PROJ"

mkdir -p logs/cordblood/slurm

MANIFEST=scripts/cordblood/manifest_tier1.tsv
TASK=$SLURM_ARRAY_TASK_ID

# +2 = skip header (1) + convert 0-based to 1-based for sed
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
    echo "[tier1 task $TASK] method=$METHOD emb=$EMBEDDING seed=$SEED"
    echo "  outdir=$OUTDIR"
    echo "  host=$(hostname)  $(date)"
    echo "==================================================================="
} | tee -a "$LOG"

# Stage data to node-local TMPDIR to avoid network-FS contention
LOCAL_DATA="$TMPDIR/cordblood_addpop.h5ad"
cp "$PROJ/data/cordblood_addpop.h5ad" "$LOCAL_DATA"

scripts/cordblood/dispatch_task.sh "$METHOD" "$EMBEDDING" "$SEED" "$OUTDIR" "$LOCAL_DATA" 2>&1 | tee -a "$LOG"
echo "[tier1 task $TASK] EXIT $?" | tee -a "$LOG"
