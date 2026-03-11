#!/bin/bash
# Evaluate SF2M on Klein dataset: PCA-30 and DM-10 configurations

set -e
cd "$(dirname "$0")/../.."

echo "=== SF2M Eval: PCA-30 ==="
python scripts/sf2m/03_evaluate.py \
    --model_dir logs/sf2m/pca30/model \
    --F_obs_path data/klein/F_obs.csv \
    --eval_mode both \
    --n_sims 10 \
    --n_sims_fate 100 \
    --sde_steps 400 \
    --k 15 \
    --gpu 0

echo "=== SF2M Eval: DM-10 ==="
python scripts/sf2m/03_evaluate.py \
    --model_dir logs/sf2m/dm10/model \
    --F_obs_path data/klein/F_obs.csv \
    --eval_mode both \
    --n_sims 10 \
    --n_sims_fate 100 \
    --sde_steps 400 \
    --k 15 \
    --gpu 0

echo "Done."
