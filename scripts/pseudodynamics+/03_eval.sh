#!/bin/bash
# pseudodynamics+ W2 sweep across configs × sim_fn × t_end_norm × noise_scale.
# Each (config, sim_fn, t, noise) combo writes a .done marker on success and
# is skipped if the marker already exists. Safe to re-launch.

set -u
# PY="${PY:-/rds/user/wz369/hpc-work/LIBS/mamba/envs/PINN_env/bin/python}"
# PY="singularity exec /rds/user/wz369/hpc-work/containers/pseudodynamics+_torch2.10.0+cu128_20260304.sif python"
PY="${PY:-/local/scratch/wz369/PINN_env/bin/python}"
PDP_DIR="${PDP_DIR:-/rds/user/wz369/hpc-work/pseudodynamics_plus}"
EVAL_PY="${EVAL_PY:-/rds/user/wz369/hpc-work/PINN_dynamics/scripts/pseudodynamics+/03_eval_pdp_w2.py}"

cd "${PDP_DIR}"

# Markers live next to results so they survive re-runs.
marker_for() {
    local config="$1" sim_fn="$2" t="$3" noise="$4"
    local name=$(basename "$(dirname "$(dirname "${config}")")")
    local marker_dir="${PDP_DIR}/results/pseudodynamics+/${name}/_markers"
    mkdir -p "${marker_dir}"
    echo "${marker_dir}/${sim_fn}_t${t}_n${noise}.done"
}

run_one() {
    local config="$1" sim_fn="$2" t="$3" noise="$4"
    local marker; marker=$(marker_for "${config}" "${sim_fn}" "${t}" "${noise}")
    local tag="$(basename $(dirname $(dirname ${config})))/${sim_fn}/t${t}/n${noise}"

    if [ -f "${marker}" ]; then
        echo "[SKIP]   ${tag}     marker=${marker}"
        return 0
    fi

    local t0=$(date +%s)
    echo "[START]  ${tag}     $(date '+%F %T')"
    if [ "${sim_fn}" = "ode" ]; then
        ${PY} "${EVAL_PY}" \
            --sim_fn ode \
            --config_path "${PDP_DIR}/${config}" \
            --t_end_norm "${t}" \
            --device cuda:0
    else
        ${PY} "${EVAL_PY}" \
            --sim_fn "${sim_fn}" \
            --config_path "${PDP_DIR}/${config}" \
            --t_end_norm "${t}" \
            --noise_scale "${noise}" \
            --device cuda:0
    fi
    local rc=$?
    local elapsed=$(( $(date +%s) - t0 ))
    if [ $rc -eq 0 ]; then
        date '+%F %T' > "${marker}"
        echo "[DONE]   ${tag}     exit=0  elapsed=${elapsed}s"
    else
        echo "[FAIL]   ${tag}     exit=${rc}  elapsed=${elapsed}s"
    fi
    return $rc
}

run_evaluation() {
    local config="$1"
    if [ ! -f "${PDP_DIR}/${config}" ]; then
        echo "[MISSING] config not found: ${PDP_DIR}/${config}"
        return 0
    fi
    for time in 2.5 3.0 3.5 4.0; do
        # ode: noise unused, single value
        run_one "${config}" ode "${time}" 1.0
        for noise in 0.5 1 1.5; do
            run_one "${config}" sde  "${time}" "${noise}"
            run_one "${config}" sb   "${time}" "${noise}"
        done
    done
}

# === DM configs ===
# Non OT-assumption
# run_evaluation "logs/klein_DM_10_lD10_lv1_lgNone/pde_params_tsense/V0_config.json" &
# run_evaluation "logs/klein_DM_10_lD1_lv10_lgNone/pde_params_tsense/V0_config.json" &
# run_evaluation "logs/klein_DM_10_lD1_lv1_lgNone/pde_params_tsense/V0_config.json" &
# wait

# # OT-assumption
# run_evaluation "logs/klein_DM_10_lD1_cfm1_lgNone_b512/pde_params_tsense/V0_config.json" &
# run_evaluation "logs/klein_DM_10_lD1_cfm2_lgNone_b512/pde_params_tsense/V0_config.json" &
# wait

# run_evaluation "logs/klein_DM_10_lD1_cfm1_lv1_lgNone_b512/pde_params_tsense/V0_config.json" &
# run_evaluation "logs/klein_DM_10_lD1_cfm10_lgNone_b512/pde_params_tsense/V0_config.json" &
# run_evaluation "logs/klein_DM_10_lD1_cfm10_lgNone_b1024/pde_params_tsense/V0_config.json" &
# wait

# DMscaled OT-assumption
# run_evaluation "logs/klein_DMscaled_10_cfm5_b512/pde_params_tsense/V0_config.json" &
# run_evaluation "logs/klein_DMscaled_10_cfm5_b1024/pde_params_tsense/V0_config.json" &
# wait

# run_evaluation "logs/klein_DMscaled_10_cfm10_b512/pde_params_tsense/V0_config.json" &
# run_evaluation "logs/klein_DMscaled_10_cfm10_b1024/pde_params_tsense/V0_config.json" &
# wait

# === PC configs ===
# No OT-assumption
run_evaluation "logs/klein_PC_30_lD1_lv1_lgNone/pde_params_tsense/V0_config.json" &

# OT-assumption
run_evaluation "logs/klein_PC30_lD1_cfm1_lgNone/pde_params_tsense/V0_config.json" &
run_evaluation "logs/klein_PC30_lD1_cfm1_lv1_lgNone/pde_params_tsense/V0_config.json" &
wait

run_evaluation "logs/klein_PC30_lD1_cfm2_lgNone/pde_params_tsense/V0_config.json" &
run_evaluation "logs/klein_PC30_lD1_cfm10_lgNone/pde_params_tsense/V0_config.json" &
run_evaluation "logs/klein_PC30_lD1_cfm10_lgNone_b1024/pde_params_tsense/V0_config.json" &
wait

echo "[ALL]    pdp+ sweep finished at $(date '+%F %T')"
