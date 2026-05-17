#!/bin/bash
#SBATCH --job-name=prescient_pc30_sd_sweep
#SBATCH --account=GOTTGENS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/rds/user/wz369/hpc-work/PINN_dynamics/logs/PRESCIENT/sd_sweep_%j.out
#SBATCH --error=/rds/user/wz369/hpc-work/PINN_dynamics/logs/PRESCIENT/sd_sweep_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wz369@cam.ac.uk
#
# Sweep PRESCIENT PC30 sim noise (config.train_sd) at evaluation time only.
# Background: PC30 has highest Pearson r (~0.90) and fate accuracy (~0.52) of the
# cluster but the worst W2 (~17.7 vs cluster 10.5–12.7). The drift quality is
# excellent; the inflated W2 is the SDE cloud spread driven by train_sd=0.5 on
# X_pca (std≈2 → noise step ≈25% of data std per step).
#
# train_sd is NOT in the model weights — fate_eval_pipeline.py:379 builds
# `z = randn * config.train_sd` external to the network at every SDE step. We
# can mutate config.train_sd at eval time (no retraining) to see the trade-off.
#
# Per-(seed, sd) results land in:
#   results/PRESCIENT/pca_run/growth_weights-softplus_1_500-1e-06/seed_<S>/
#       eval_sim_sd_<sd>/eval_combined.csv
# (the default eval_combined.csv at the seed root is preserved)

set -euo pipefail

cd /rds/user/wz369/hpc-work/PINN_dynamics
mkdir -p logs/PRESCIENT

PYTHON="singularity exec --nv /rds/user/wz369/hpc-work/containers/flow.sif python"
DATA="data/klein_addpop.h5ad"
MODEL_ROOT="results/PRESCIENT/pca_run/growth_weights-softplus_1_500-1e-06"
SEEDS=(0 2 42)
SDS=(0.05 0.1 0.25)

echo "[JOB] prescient_pc30_sd_sweep  job=${SLURM_JOB_ID:-local}  start=$(date '+%F %T')"
nvidia-smi -L || true

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
        # Wrap in subshell + || true so a single eval failure does not abort
        # the entire sweep under `set -e`. Failures are logged for triage.
        if ! $PYTHON scripts/prescient/03_evaluate.py \
                --model_dir "$model_dir" \
                --data_path "$DATA" \
                --obsm_key  X_pca \
                --n_dims    30 \
                --sim_sd    "$sd" \
                --device    cuda:0; then
            echo "[ERROR] seed=${seed} sd=${sd} eval failed (rc=$?). Continuing sweep."
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
