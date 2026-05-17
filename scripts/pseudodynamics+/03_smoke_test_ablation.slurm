#!/bin/bash
#SBATCH --job-name=pdp_w2_smoke
#SBATCH --account=GOTTGENS-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/rds/user/wz369/hpc-work/pseudodynamics_plus/logs/eval_klein/pdp_w2_smoke_%j.out
#SBATCH --error=/rds/user/wz369/hpc-work/pseudodynamics_plus/logs/eval_klein/pdp_w2_smoke_%j.err

# Smoke-test: verify 03_eval_pdp_w2.py produces a W2 number on
# (klein_OT/baseline/seed_0) and (klein_OT/lambdaR_1000/seed_0).
# Pass --output_path explicitly because the script's auto-naming uses
# split("/")[-2] which under the ablation layout returns "seed_0" and
# collides across arms.

set -u
PDP_DIR=/rds/user/wz369/hpc-work/pseudodynamics_plus
PY=${PY:-/rds/user/wz369/hpc-work/LIBS/mamba/envs/PINN_env/bin/python}
EVAL_PY=/rds/user/wz369/hpc-work/pseudodynamics_plus/scripts/pseudodynamics+/03_eval_pdp_w2.py
OUT_DIR=${PDP_DIR}/results/pseudodynamics+/_smoke_test_ablation
mkdir -p "${OUT_DIR}" "${PDP_DIR}/logs/eval_klein"

cd "${PDP_DIR}"
echo "[JOB] pdp_w2_smoke job=${SLURM_JOB_ID:-local} node=$(hostname) start=$(date '+%F %T')"
nvidia-smi -L || true

run_one() {
    local cfg="$1" tag="$2"
    echo "════════════════════════════════════════════════════════════════════"
    echo "[SMOKE] ${tag}"
    echo "[CFG]   ${cfg}"
    echo "════════════════════════════════════════════════════════════════════"
    local t0=$(date +%s)
    ${PY} ${EVAL_PY} \
        --sim_fn ode \
        --config_path "${cfg}" \
        --t_end_norm 2.5 \
        --output_path "${OUT_DIR}/${tag}_t2.5_ode_w2.csv" \
        --device cuda:0
    local rc=$?
    echo "[SMOKE] ${tag} rc=${rc} elapsed=$(( $(date +%s) - t0 ))s"
    return $rc
}

run_one "${PDP_DIR}/logs/klein_OT_ablation/baseline/seed_0/pde_params_tsense/V28615539_config.json" \
        "klein_OT_baseline_seed0"

run_one "${PDP_DIR}/logs/klein_OT_ablation/lambdaR_1000/seed_0/pde_params_tsense/V28886826_config.json" \
        "klein_OT_lambdaR_1000_seed0"

echo "[JOB] pdp_w2_smoke end=$(date '+%F %T')"
ls -la "${OUT_DIR}/"
