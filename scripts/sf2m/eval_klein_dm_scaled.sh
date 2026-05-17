#!/bin/bash
# Evaluate SF2M on Klein dataset: DM_EigenVectors_scaled models (dm10_scaled)
# Identical to eval_klein.sh but targets logs/sf2m/dm10_scaled/ models.
# Each (repeat) combo writes a .done marker and is skipped on re-submit.

cd "$(dirname "$0")/../.."

PY="singularity exec --nv /rds/user/wz369/hpc-work/containers/flow.sif  python"
DEVICE="${DEVICE:-cuda:0}"

REPEATS=(model_r1 model_r2 model_r3)

run_one() {
    local rep="$1"
    local model_dir="logs/sf2m/dm10_scaled/${rep}"
    local marker="${model_dir}/eval_klein.done"
    local tag="sf2m/dm10_scaled/${rep}"

    if [ -f "${marker}" ]; then
        echo "[SKIP]   ${tag}     marker=${marker}"
        return 0
    fi
    if [ ! -d "${model_dir}" ]; then
        echo "[MISSING] ${tag}    model_dir not found: ${model_dir}"
        return 0
    fi

    local t0=$(date +%s)
    echo "[START]  ${tag}     $(date '+%F %T')"
    $PY scripts/sf2m/03_evaluate.py \
        --model_dir "${model_dir}" \
        --F_obs_path data/klein/F_obs.csv \
        --sim_modes ode sde \
        --n_sims_fate 100 \
        --n_sims_w2 10 \
        --sde_steps 400 \
        --k 15 \
        --device "${DEVICE}"
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

for REP in "${REPEATS[@]}"; do
    run_one "${REP}"
done

echo "[ALL]    sf2m dm10_scaled sweep finished at $(date '+%F %T')"
