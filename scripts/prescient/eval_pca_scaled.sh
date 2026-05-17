#!/bin/bash
# Evaluate PRESCIENT pca_scaled_run (seeds 1, 11, 22)
# Uses adata.uns['PC_scaler'] to invert endpoints back to raw PC space for w2_raw.
#
# Usage:  bash scripts/prescient/eval_pca_scaled.sh

set -euo pipefail
cd /rds/user/wz369/hpc-work/PINN_dynamics

PYTHON="singularity exec --nv /rds/user/wz369/hpc-work/containers/flow.sif python"
DATA="data/klein_addpop.h5ad"
MODEL_ROOT="results/PRESCIENT/pca_scaled_run/PCA_SCALED-softplus_1_500-1e-06"

for seed in 1 11 22; do
    model_dir="${MODEL_ROOT}/seed_${seed}"
    if [ -f "${model_dir}/eval_combined.csv" ]; then
        echo "[SKIP] seed=${seed} already evaluated"
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

echo "=== Results ==="
for seed in 1 11 22; do
    f="${MODEL_ROOT}/seed_${seed}/eval_combined.csv"
    if [ -f "$f" ]; then echo "  seed=${seed}:"; cat "$f"; fi
done
