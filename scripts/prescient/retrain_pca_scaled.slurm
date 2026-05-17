#!/bin/bash
#SBATCH --job-name=prescient_pca_scaled
#SBATCH --account=GOTTGENS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/rds/user/wz369/hpc-work/PINN_dynamics/logs/PRESCIENT/slurm_pca_scaled_%j.out
#SBATCH --error=/rds/user/wz369/hpc-work/PINN_dynamics/logs/PRESCIENT/slurm_pca_scaled_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wz369@cam.ac.uk
#
# Retrain PRESCIENT on X_pca_scaled (30D, std≈1 in adata.obsm).
# Background: the previous pca_run trained on raw X_pca (std≈2) with
# train_sd=0.5 → SDE cloud over-spread → w2_raw=17.7 vs cluster 10-12.
# Retraining on X_pca_scaled (std≈1) puts the noise scale in a better regime.
# Eval reads adata.uns['PC_scaler'] to invert endpoints back to raw PC units.
#
# Pipeline (3 seeds): 01_prepare_data.py → 02_train.py → 03_evaluate.py
#
# Outputs:
#   results/PRESCIENT/pca_scaled_run/
#     ├── data.pt
#     ├── growth_weights.pt
#     ├── PCA_SCALED-softplus_1_500-1e-06/seed_{0,2,42}/
#     │      ├── train.best.pt
#     │      ├── config.pt
#     │      ├── eval_combined.csv
#     │      ├── F_hat_sde.csv
#     │      └── w2_per_clone_sde.csv

set -euo pipefail

cd /rds/user/wz369/hpc-work/PINN_dynamics
mkdir -p logs/PRESCIENT

PYTHON="singularity exec --nv /rds/user/wz369/hpc-work/containers/flow.sif python"
DATA="data/klein_addpop.h5ad"
OUT_ROOT="results/PRESCIENT/pca_scaled_run"
WEIGHT_NAME="PCA_SCALED"
SEEDS=(0 2 42)

echo "[JOB] prescient_pca_scaled  job=${SLURM_JOB_ID:-local}  node=$(hostname)  start=$(date '+%F %T')"
nvidia-smi -L || true

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: prepare data.pt (only if not already prepared)
# ─────────────────────────────────────────────────────────────────────────────
if [ ! -f "${OUT_ROOT}/data.pt" ]; then
    echo "[STEP] 01_prepare_data.py  →  ${OUT_ROOT}/"
    $PYTHON scripts/prescient/01_prepare_data.py \
        --data_path  "$DATA" \
        --output_dir results/PRESCIENT \
        --run_name   pca_scaled_run \
        --obsm_key   X_pca_scaled \
        --n_dims     30
else
    echo "[SKIP] ${OUT_ROOT}/data.pt already exists"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: train all seeds in parallel on the same GPU.
# Each seed sleeps a random 0–30s before launch to stagger CUDA init.
# ─────────────────────────────────────────────────────────────────────────────
train_pids=()
for seed in "${SEEDS[@]}"; do
    expected="${OUT_ROOT}/${WEIGHT_NAME}-softplus_1_500-1e-06/seed_${seed}/train.best.pt"
    if [ -f "$expected" ]; then
        echo "[SKIP] seed=${seed} already trained: $expected"
        continue
    fi
    log_file="${OUT_ROOT}/train_seed_${seed}.log"
    mkdir -p "${OUT_ROOT}"
    (
        delay=$(( RANDOM % 30 ))
        echo "[STEP] 02_train.py  seed=${seed}  (sleeping ${delay}s before launch)"
        sleep "$delay"
        $PYTHON scripts/prescient/02_train.py \
            --data_path   "${OUT_ROOT}/data.pt" \
            --out_dir     "${OUT_ROOT}" \
            --config      pca \
            --weight_name "$WEIGHT_NAME" \
            --seed        "$seed" \
            --gpu         0 \
            > "$log_file" 2>&1
        echo "[DONE] seed=${seed}  rc=$?  log=${log_file}"
    ) &
    train_pids+=($!)
done

# Wait for all parallel training jobs and report failures.
train_failed=0
for pid in "${train_pids[@]}"; do
    if ! wait "$pid"; then
        train_failed=$((train_failed + 1))
    fi
done
if [ "$train_failed" -gt 0 ]; then
    echo "[WARN] ${train_failed} seed(s) failed during training; continuing to eval anyway."
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: evaluate each seed
# ─────────────────────────────────────────────────────────────────────────────
for seed in "${SEEDS[@]}"; do
    model_dir="${OUT_ROOT}/${WEIGHT_NAME}-softplus_1_500-1e-06/seed_${seed}"
    eval_csv="${model_dir}/eval_combined.csv"
    if [ -f "$eval_csv" ]; then
        echo "[SKIP] seed=${seed} already evaluated: $eval_csv"
        continue
    fi
    if [ ! -f "${model_dir}/train.best.pt" ]; then
        echo "[WARN] seed=${seed} missing checkpoint — skipping eval"
        continue
    fi
    echo "[STEP] 03_evaluate.py  seed=${seed}"
    if ! $PYTHON scripts/prescient/03_evaluate.py \
        --model_dir "$model_dir" \
        --data_path "$DATA" \
        --obsm_key  X_pca_scaled \
        --n_dims    30 \
        --device    cuda:0; then
        echo "[ERROR] eval failed for seed=${seed}"
    fi
done

echo "[JOB] prescient_pca_scaled  end=$(date '+%F %T')"
echo "[RESULTS] eval_combined.csv per seed:"
for seed in "${SEEDS[@]}"; do
    f="${OUT_ROOT}/${WEIGHT_NAME}-softplus_1_500-1e-06/seed_${seed}/eval_combined.csv"
    if [ -f "$f" ]; then echo "  seed=${seed}:"; cat "$f"; fi
done
