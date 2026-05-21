#!/bin/bash
#SBATCH --job-name=eps_sweep
#SBATCH --account=GOTTGENS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/rds/user/wz369/hpc-work/pseudodynamics_plus/logs/eps_sweep_%j.out
#SBATCH --error=/rds/user/wz369/hpc-work/pseudodynamics_plus/logs/eps_sweep_%j.err

set -u
cd /rds/user/wz369/hpc-work/pseudodynamics_plus
mkdir -p logs results/epsilon_sensitivity

PYTHON="singularity exec --nv \
    --bind /rds/user/wz369/hpc-work/pseudodynamics_plus:/work \
    --bind /rds/user/wz369/hpc-work/PINN_dynamics:/rds/user/wz369/hpc-work/PINN_dynamics \
    --bind /rds/user/wz369/hpc-work/PINN_dynamics:/PINN_dynamics \
    --bind /rds/user/wz369/hpc-work/pseudodynamics_plus/src/pseudodynamics:/usr/local/lib/python3.10/dist-packages/pseudodynamics \
    /rds/user/wz369/hpc-work/containers/pseudodynamics+_torch2.10.0+cu128_20260304.sif python"

echo "[`date`] starting ε-sensitivity sweep"
${PYTHON} scripts/pseudodynamics+/10_epsilon_sensitivity.py "$@"
echo "[`date`] done"
