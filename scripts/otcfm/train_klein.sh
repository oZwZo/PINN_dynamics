#!/bin/bash
# Train OT-CFM on Klein dataset: PCA-30 and DM-10 configurations

set -e
# cd "$(dirname "$0")/../.."
cd /rds/user/wz369/hpc-work/PINN_dynamics
python="singularity exec --nv /rds/user/wz369/hpc-work/containers/flow.sif python"
DATA_PATH="data/klein_addpop.h5ad"

echo "=== OT-CFM: PCA-30 ==="
$python scripts/otcfm/02_train.py \
    --data_path "$DATA_PATH" \
    --obsm_key X_pca_scaled \
    --n_dims 30 \
    --width 128 \
    --save_dir logs/otcfm/pca30/model_r1 \
    --niters 100000 \
    --batch_size 256 \
    --sigma 0.1 \
    --lr 1e-4 \
    --seed 42 \
    --gpu 0 \
    --val_freq 1000 \
    --patience 10000 \
    --val_frac 0.1  &

$python scripts/otcfm/02_train.py \
    --data_path "$DATA_PATH" \
    --obsm_key X_pca_scaled \
    --n_dims 30 \
    --width 128 \
    --save_dir logs/otcfm/pca30/model_r2 \
    --niters 100000 \
    --batch_size 256 \
    --sigma 0.1 \
    --lr 1e-4 \
    --seed 42 \
    --gpu 0 \
    --val_freq 1000 \
    --patience 10000 \
    --val_frac 0.1  &

$python scripts/otcfm/02_train.py \
    --data_path "$DATA_PATH" \
    --obsm_key X_pca_scaled \
    --n_dims 30 \
    --width 128 \
    --save_dir logs/otcfm/pca30/model_r3 \
    --niters 100000 \
    --batch_size 256 \
    --sigma 0.1 \
    --lr 1e-4 \
    --seed 42 \
    --gpu 0 \
    --val_freq 1000 \
    --patience 10000 \
    --val_frac 0.1  &

wait

echo "=== OT-CFM: DM-10 ==="
$python scripts/otcfm/02_train.py \
    --data_path "$DATA_PATH" \
    --obsm_key DM_EigenVectors \
    --n_dims 10 \
    --width 128 \
    --save_dir logs/otcfm/dm10/model_r1 \
    --niters 100000 \
    --batch_size 256 \
    --sigma 0.1 \
    --lr 1e-4 \
    --seed 42 \
    --gpu 0 \
    --val_freq 1000 \
    --patience 10000 \
    --val_frac 0.1 &

$python scripts/otcfm/02_train.py \
    --data_path "$DATA_PATH" \
    --obsm_key DM_EigenVectors \
    --n_dims 10 \
    --width 128 \
    --save_dir logs/otcfm/dm10/model_r2 \
    --niters 100000 \
    --batch_size 256 \
    --sigma 0.1 \
    --lr 1e-4 \
    --seed 42 \
    --gpu 0 \
    --val_freq 1000 \
    --patience 10000 \
    --val_frac 0.1 &

$python scripts/otcfm/02_train.py \
    --data_path "$DATA_PATH" \
    --obsm_key DM_EigenVectors \
    --n_dims 10 \
    --width 128 \
    --save_dir logs/otcfm/dm10/model_r3 \
    --niters 100000 \
    --batch_size 256 \
    --sigma 0.1 \
    --lr 1e-4 \
    --seed 42 \
    --gpu 0 \
    --val_freq 1000 \
    --patience 10000 \
    --val_frac 0.1 &

echo "Done."
