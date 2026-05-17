#!/bin/bash
# Train + eval PRESCIENT on X_pca_scaled (30D, std≈1) with 3 seeds
# Eval reads adata.uns['PC_scaler'] to invert back to raw PC units.
#
# Usage:  bash scripts/prescient/train_pca_scaled.sh   (on GPU node)

set -euo pipefail

cd /rds/user/wz369/hpc-work/PINN_dynamics

PYTHON="singularity exec --nv /rds/user/wz369/hpc-work/containers/flow.sif python"
DATA="data/klein_addpop.h5ad"
OUT_ROOT="results/PRESCIENT/pca_scaled_run"
WEIGHT_NAME="PCA_SCALED"
SEEDS=(1 11 22)

echo "=== PRESCIENT: X_pca_scaled (seeds ${SEEDS[*]}) ==="

# ── Step 1: prepare data.pt ──────────────────────────────────────────────────
if [ ! -f "${OUT_ROOT}/data.pt" ]; then
    echo "[STEP] 01_prepare_data.py → ${OUT_ROOT}/"
    $PYTHON scripts/prescient/01_prepare_data.py \
        --data_path  "$DATA" \
        --output_dir results/PRESCIENT \
        --run_name   pca_scaled_run \
        --obsm_key   X_pca_scaled \
        --n_dims     30
else
    echo "[SKIP] ${OUT_ROOT}/data.pt already exists"
fi

# ── Step 2: train 3 seeds in parallel ────────────────────────────────────────
train_pids=()
for seed in "${SEEDS[@]}"; do
    expected="${OUT_ROOT}/${WEIGHT_NAME}-softplus_1_500-1e-06/seed_${seed}/train.best.pt"
    if [ -f "$expected" ]; then
        echo "[SKIP] seed=${seed} already trained"
        continue
    fi
    log_file="${OUT_ROOT}/train_seed_${seed}.log"
    mkdir -p "${OUT_ROOT}"
    (
        delay=$(( RANDOM % 30 ))
        echo "[START] seed=${seed} (sleep ${delay}s)"
        sleep "$delay"
        $PYTHON scripts/prescient/02_train.py \
            --data_path   "${OUT_ROOT}/data.pt" \
            --out_dir     "${OUT_ROOT}" \
            --config      pca \
            --weight_name "$WEIGHT_NAME" \
            --seed        "$seed" \
            --gpu         0 \
            > "$log_file" 2>&1
        echo "[DONE] seed=${seed}"
    ) &
    train_pids+=($!)
done

wait "${train_pids[@]}"
echo "=== Training complete ==="

# ── Step 3: evaluate each seed ───────────────────────────────────────────────
for seed in "${SEEDS[@]}"; do
    model_dir="${OUT_ROOT}/${WEIGHT_NAME}-softplus_1_500-1e-06/seed_${seed}"
    if [ -f "${model_dir}/eval_combined.csv" ]; then
        echo "[SKIP] seed=${seed} already evaluated"
        continue
    fi
    if [ ! -f "${model_dir}/train.best.pt" ]; then
        echo "[WARN] seed=${seed} missing checkpoint — skipping"
        continue
    fi
    echo "[EVAL] seed=${seed}"
    $PYTHON scripts/prescient/03_evaluate.py \
        --model_dir "$model_dir" \
        --data_path "$DATA" \
        --obsm_key  X_pca_scaled \
        --n_dims    30 \
        --device    cuda:0
done

echo "=== Done ==="
for seed in "${SEEDS[@]}"; do
    f="${OUT_ROOT}/${WEIGHT_NAME}-softplus_1_500-1e-06/seed_${seed}/eval_combined.csv"
    if [ -f "$f" ]; then echo "  seed=${seed}:"; cat "$f"; fi
done
