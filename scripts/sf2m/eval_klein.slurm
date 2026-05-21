#!/bin/bash
#SBATCH --job-name=sf2m_otcfm_eval
#SBATCH --account=GOTTGENS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/rds/user/wz369/hpc-work/PINN_dynamics/logs/eval_klein/sf2m_otcfm_%j.out
#SBATCH --error=/rds/user/wz369/hpc-work/PINN_dynamics/logs/eval_klein/sf2m_otcfm_%j.err

# Resumable W2/fate evaluation sweep for sf2m AND otcfm.
# Each (method, embedding, repeat) combo writes a .done marker and is
# skipped on re-submit. To force a re-run: delete the marker.
#
# Inspect progress next morning:
#   grep '\[DONE\]\|\[FAIL\]\|\[SKIP\]\|\[MISSING\]' logs/eval_klein/sf2m_otcfm_<jobid>.out
#   tail -F logs/eval_klein/sf2m_otcfm_<jobid>.out

set -u
cd /rds/user/wz369/hpc-work/PINN_dynamics

export DEVICE="cuda:0"

echo "[JOB]    sf2m_otcfm_eval  job=${SLURM_JOB_ID:-local}  node=$(hostname)  start=$(date '+%F %T')"
nvidia-smi -L || true

echo "============================================================"
echo "  sf2m sweep"
echo "============================================================"
bash scripts/sf2m/eval_klein.sh
sf2m_rc=$?

echo "============================================================"
echo "  otcfm sweep"
echo "============================================================"
bash scripts/otcfm/eval_klein.sh
otcfm_rc=$?

echo "[JOB]    sf2m_otcfm_eval  end=$(date '+%F %T')  sf2m_rc=${sf2m_rc}  otcfm_rc=${otcfm_rc}"
