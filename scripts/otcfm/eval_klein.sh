#!/bin/bash
# Evaluate OT-CFM on Klein dataset: PCA-30 and DM-10 configurations

set -e
cd "$(dirname "$0")/../.."

echo "=== OT-CFM Eval: PCA-30 ==="
python scripts/otcfm/03_evaluate.py \
    --model_dir logs/otcfm/pca30/model \
    --F_obs_path data/klein/F_obs.csv \
    --n_sims 10 \
    --k 15 \
    --gpu 0

echo "=== OT-CFM Eval: DM-10 ==="
python scripts/otcfm/03_evaluate.py \
    --model_dir logs/otcfm/dm10/model \
    --F_obs_path data/klein/F_obs.csv \
    --n_sims 10 \
    --k 15 \
    --gpu 0

echo "Done."
