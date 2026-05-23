#!/bin/bash
# Run 3 scDiffEq trainings on the jupyter-gpu allocation (job 29539734, gpu-q-85).
# Staggered 90s apart to avoid Lightning's accelerator.setup_device race.
# Reduced epochs=500 to fit within ~5h budget on one A100 shared with jupyter.

set -uo pipefail   # no -e so one seed failure doesn't kill the others

PROJ=/rds/user/wz369/hpc-work/PINN_dynamics
SCDIFFEQ_DIR=/rds/user/wz369/hpc-work/scDiffEq
SCRIPT="$PROJ/scripts/scdiffeq/run_scdiffeq_cordblood.py"
EVAL_SCRIPT="$PROJ/scripts/scdiffeq/evaluate_scdiffeq_cordblood.py"
DATA="$PROJ/data/cordblood_addpop.h5ad"
F_OBS="$PROJ/data/CordBlood/F_obs_day17.csv"
CLONE_PROP="$PROJ/data/CordBlood/clone_proportions_day17.csv"
LOGDIR="$PROJ/logs/cordblood/scdiffeq"

mkdir -p "$LOGDIR"

# (config, seed) triples — one DM-9 + two PCA-30
RUNS=("dm9 42" "pca30 42" "pca30 7")
EPOCHS=500
TAG=jupyter

train_eval_one() {
    local cfg="$1" seed="$2"
    local log="$LOGDIR/scdiffeq_cb_${cfg}_s${seed}-${TAG}.log"
    local outdir="$LOGDIR/${cfg}_fp_vr_s${seed}"
    mkdir -p "$outdir"
    echo "[$(date)] START cfg=$cfg seed=$seed epochs=$EPOCHS log=$log"
    uv run --directory "$SCDIFFEQ_DIR" python "$SCRIPT" \
        --config "$cfg" --data "$DATA" --output_dir "$LOGDIR" \
        --epochs $EPOCHS --seed "$seed" > "$log" 2>&1

    local ckpt
    ckpt=$(find "$outdir" -name "*.ckpt" ! -name "last.ckpt" | sort | tail -1)
    [[ -z "$ckpt" ]] && ckpt=$(find "$outdir" -name "last.ckpt" | head -1)
    if [[ -n "$ckpt" ]]; then
        uv run --directory "$SCDIFFEQ_DIR" python "$EVAL_SCRIPT" \
            --config "$cfg" --data "$DATA" --ckpt_path "$ckpt" \
            --f_obs "$F_OBS" --clone_proportions "$CLONE_PROP" \
            --output_dir "$outdir/eval" --seed "$seed" >> "$log" 2>&1
    else
        echo "[WARN] no ckpt under $outdir, skipping eval" >> "$log"
    fi
    echo "[$(date)] DONE  cfg=$cfg seed=$seed"
}

pids=()
i=0
for spec in "${RUNS[@]}"; do
    read -r cfg seed <<< "$spec"
    if (( i > 0 )); then
        echo "[$(date)] sleep 90s before launching cfg=$cfg seed=$seed (stagger)"
        sleep 90
    fi
    train_eval_one "$cfg" "$seed" &
    pids+=($!)
    i=$((i+1))
done
echo "[$(date)] PIDs: ${pids[*]}"
for pid in "${pids[@]}"; do
    wait "$pid" || echo "[$(date)] pid=$pid exited non-zero"
done
echo "[$(date)] All 3 scDiffEq jupyter runs finished."
