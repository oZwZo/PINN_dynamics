#!/bin/bash
#SBATCH --job-name=sf2m_dm_scaled
#SBATCH --account=GOTTGENS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/rds/user/wz369/hpc-work/PINN_dynamics/logs/sf2m/slurm_dm_scaled_%j.out
#SBATCH --error=/rds/user/wz369/hpc-work/PINN_dynamics/logs/sf2m/slurm_dm_scaled_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wz369@cam.ac.uk

# SF2M DM_EigenVectors_scaled: train 3 seeds + evaluate
#
# Key design choices:
#   - obsm_key = DM_EigenVectors_scaled (pre-scaled in adata, std≈1)
#   - --no_standardize (data already at unit scale)
#   - sigma = 0.1 (matches OTCFM; better for unit-scale data)
#   - Seeds 0, 2, 42 for variance estimate
#   - Eval uses adata.uns['DM_scaler'] for inverse → w2_raw in raw DM units
#
# Existing models at logs/sf2m/dm10/ are UNTOUCHED.

set -eu
cd /rds/user/wz369/hpc-work/PINN_dynamics

PY="singularity exec --nv /rds/user/wz369/hpc-work/containers/flow.sif python"
DATA_PATH="data/klein_addpop.h5ad"

echo "[JOB] sf2m_dm_scaled  job=${SLURM_JOB_ID:-local}  node=$(hostname)  start=$(date '+%F %T')"
nvidia-smi -L || true

# ═══════════════════════════════════════════════════════════════
# STEP 1: Train 3 seeds (sequential — each takes ~30-60 min)
# ═══════════════════════════════════════════════════════════════
SEEDS=(0 2 42)
REPS=(model_r1 model_r2 model_r3)

echo "============================================================"
echo "  STEP 1: Training"
echo "============================================================"

for i in 0 1 2; do
    seed=${SEEDS[$i]}
    rep=${REPS[$i]}
    save_dir="logs/sf2m/dm10_scaled/${rep}"

    if [ -f "${save_dir}/ckpt.pt" ]; then
        echo "[SKIP] ${rep} (seed=${seed}) — checkpoint exists"
        continue
    fi

    echo "[TRAIN] ${rep} seed=${seed} → ${save_dir}"
    $PY scripts/sf2m/02_train.py \
        --data_path "$DATA_PATH" \
        --obsm_key DM_EigenVectors_scaled \
        --n_dims 10 \
        --width 128 \
        --save_dir "${save_dir}" \
        --niters 100000 \
        --batch_size 256 \
        --sigma 0.1 \
        --lr 1e-4 \
        --seed "${seed}" \
        --gpu 0 \
        --no_standardize \
        --val_freq 1000 \
        --patience 10000 \
        --val_frac 0.1
    echo "[DONE] ${rep} seed=${seed}"
done

# ═══════════════════════════════════════════════════════════════
# STEP 2: Evaluate all 3 seeds
# ═══════════════════════════════════════════════════════════════
echo "============================================================"
echo "  STEP 2: Evaluation"
echo "============================================================"

export DEVICE="cuda:0"
bash scripts/sf2m/eval_klein_dm_scaled.sh
eval_rc=$?

echo "[JOB] sf2m_dm_scaled  end=$(date '+%F %T')  eval_rc=${eval_rc}"
