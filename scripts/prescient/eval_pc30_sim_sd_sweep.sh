#!/bin/bash
# Sweep PRESCIENT PC30 sim noise (config.train_sd) at evaluation time only.
# No retraining. Each (seed, sd) writes to:
#   results/PRESCIENT/pca_run/growth_weights-softplus_1_500-1e-06/seed_<S>/eval_sim_sd_<sd>/eval_combined.csv
#
# Usage:
#   bash scripts/prescient/eval_pc30_sim_sd_sweep.sh
#
# Note: requires a CUDA-capable node. The eval entropic-OT W2 (~2k × 100 sims)
# OOMs on CPU, and KeOps silently falls back to CPU when CUDA is unavailable.
# Auto-detects device; aborts early on a CPU-only node with a clear message.

set -uo pipefail

cd /rds/user/wz369/hpc-work/PINN_dynamics
mkdir -p logs/PRESCIENT

PYTHON="singularity exec --nv /rds/user/wz369/hpc-work/containers/flow.sif python"
DATA="data/klein_addpop.h5ad"
MODEL_ROOT="results/PRESCIENT/pca_run/growth_weights-softplus_1_500-1e-06"
SEEDS=(0 2 42)
SDS=(0.05 0.1 0.25)

echo "[JOB] prescient_pc30_sd_sweep  start=$(date '+%F %T')  host=$(hostname)"

# ── Device auto-detect ────────────────────────────────────────────────────────
DEVICE=$($PYTHON -c "import torch; print('cuda:0' if torch.cuda.is_available() else 'cpu')" 2>/dev/null)
echo "[INFO] auto-detected device: ${DEVICE}"
if [ "$DEVICE" = "cpu" ]; then
    echo "[ABORT] no CUDA available on this node. The W2 entropic-OT step (sinkhorn"
    echo "        over ~2k cells × 100 sims) will OOM on CPU. Submit on a GPU node."
    exit 1
fi
nvidia-smi -L || true

# ── Sweep ─────────────────────────────────────────────────────────────────────
for seed in "${SEEDS[@]}"; do
    model_dir="${MODEL_ROOT}/seed_${seed}"
    if [ ! -f "${model_dir}/train.best.pt" ]; then
        echo "[WARN] missing checkpoint: ${model_dir}/train.best.pt — skipping seed=${seed}"
        continue
    fi
    for sd in "${SDS[@]}"; do
        out_dir="${model_dir}/eval_sim_sd_${sd}"
        eval_csv="${out_dir}/eval_combined.csv"
        if [ -f "$eval_csv" ]; then
            echo "[SKIP] seed=${seed} sd=${sd} already evaluated: $eval_csv"
            continue
        fi
        echo "[RUN] seed=${seed} sd=${sd}"
        if ! $PYTHON scripts/prescient/03_evaluate.py \
                --model_dir "$model_dir" \
                --data_path "$DATA" \
                --obsm_key  X_pca \
                --n_dims    30 \
                --sim_sd    "$sd" \
                --device    "$DEVICE"; then
            rc=$?
            echo "[ERROR] seed=${seed} sd=${sd} eval failed (rc=${rc}). Continuing sweep."
        fi
    done
done

echo "[JOB] prescient_pc30_sd_sweep  end=$(date '+%F %T')"
echo "[RESULTS] eval_combined.csv per (seed, sd):"
for seed in "${SEEDS[@]}"; do
    for sd in "${SDS[@]}"; do
        f="${MODEL_ROOT}/seed_${seed}/eval_sim_sd_${sd}/eval_combined.csv"
        if [ -f "$f" ]; then echo "  seed=${seed} sd=${sd}:"; cat "$f"; fi
    done
done
