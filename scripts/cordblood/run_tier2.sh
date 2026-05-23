#!/usr/bin/env bash
# Interactive runner for tier-2 (same shape as run_tier1.sh).
set -euo pipefail

PROJ=/rds/user/wz369/hpc-work/PINN_dynamics
cd "$PROJ"

MANIFEST=scripts/cordblood/manifest_tier2.tsv
DATA="$PROJ/data/cordblood_addpop.h5ad"
[[ -f "$DATA" ]] || { echo "Missing $DATA — run 01_preprocess.py first"; exit 2; }

run_row() {
    local task="$1"
    local line; line=$(sed -n "$((task + 2))p" "$MANIFEST")
    [[ -n "$line" ]] || { echo "No row $task in $MANIFEST"; return 1; }
    read -r TASK_ID METHOD EMBEDDING SEED OUTDIR <<< "$line"
    mkdir -p "$OUTDIR" "logs/cordblood/$METHOD"
    LOG="logs/cordblood/$METHOD/${EMBEDDING}_s${SEED}.log"
    echo "[interactive task $task] method=$METHOD emb=$EMBEDDING seed=$SEED" | tee -a "$LOG"
    scripts/cordblood/dispatch_task.sh "$METHOD" "$EMBEDDING" "$SEED" "$OUTDIR" "$DATA" 2>&1 | tee -a "$LOG"
}

ARG="${1:-}"
if [[ -z "$ARG" ]]; then
    echo "usage: $0 <task_id|all>"; exit 2
fi
if [[ "$ARG" == "all" ]]; then
    N=$(($(wc -l < "$MANIFEST") - 1))
    for ((i=0; i<N; i++)); do run_row "$i" || true; done
else
    run_row "$ARG"
fi
