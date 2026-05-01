#!/bin/bash
#SBATCH --job-name=pdp_w2_eval
#SBATCH --account=GOTTGENS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/rds/user/wz369/hpc-work/PINN_dynamics/logs/eval_klein/pdp_w2_%j.out
#SBATCH --error=/rds/user/wz369/hpc-work/PINN_dynamics/logs/eval_klein/pdp_w2_%j.err

# Resumable W2 sweep for pseudodynamics+ across configs × sim_fn × t × noise.
# Each (config, sim_fn, t, noise) combo writes a .done marker on success and
# is skipped on re-submit. To force a re-run: delete the marker file.
#
# Inspect progress next morning:
#   grep -E '\[DONE\]|\[FAIL\]|\[SKIP\]|\[MISSING\]' logs/eval_klein/pdp_w2_<jobid>.out
#   tail -F logs/eval_klein/pdp_w2_<jobid>.out

set -u
cd /rds/user/wz369/hpc-work/PINN_dynamics

echo "[JOB]    pdp_w2_eval  job=${SLURM_JOB_ID:-local}  node=$(hostname)  start=$(date '+%F %T')"
nvidia-smi -L || true

bash scripts/pseudodynamics+/03_eval.sh
rc=$?

echo "[JOB]    pdp_w2_eval  end=$(date '+%F %T')  rc=${rc}"
