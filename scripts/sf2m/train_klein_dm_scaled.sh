#!/bin/bash
# Train SF2M on Klein dataset: DM_EigenVectors_scaled (std≈1) with 3 different seeds
#
# Key differences from the raw DM10 run:
#   1. obsm_key = DM_EigenVectors_scaled (pre-scaled in adata, std≈1)
#   2. --no_standardize (data already at unit scale; avoid double-scaling)
#   3. Different seeds (0, 2, 42) for true variance estimate
#   4. Sigma=0.1 (matches OTCFM convention for better training on unit-scale data)
#
# Usage:
#   bash scripts/sf2m/train_klein_dm_scaled.sh        # on GPU node
#   sbatch scripts/sf2m/train_klein_dm_scaled.sh       # as SLURM job (add headers)
#
# Existing DM10 raw models at logs/sf2m/dm10/ are UNTOUCHED.
# New models land at logs/sf2m/dm10_scaled/model_r{1,2,3}.

set -e
cd /rds/user/wz369/hpc-work/PINN_dynamics
python="singularity exec --nv /rds/user/wz369/hpc-work/containers/flow.sif python"
DATA_PATH="data/klein_addpop.h5ad"

SEEDS=(0 2 42)
REPS=(model_r1 model_r2 model_r3)

echo "=== SF2M: DM_EigenVectors_scaled ==="
for i in 0 1 2; do
    seed=${SEEDS[$i]}
    rep=${REPS[$i]}
    save_dir="logs/sf2m/dm10_scaled/${rep}"

    if [ -f "${save_dir}/ckpt.pt" ]; then
        echo "[SKIP] ${rep} (seed=${seed}) — checkpoint exists at ${save_dir}/ckpt.pt"
        continue
    fi

    # Random sleep to stagger CUDA initialization and avoid GPU contention
    stagger=$((RANDOM % 30))
    echo "[START] ${rep} seed=${seed} → ${save_dir}  (sleep ${stagger}s)"
    (
        sleep "${stagger}"
        $python scripts/sf2m/02_train.py \
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
    ) &
done

wait
echo "=== Training complete ==="
