#!/bin/bash
#SBATCH -p ampere
#SBATCH -A GOTTGENS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-q-[15,64,68-75,81-82,86-90]
#SBATCH --time=4:00:00
#SBATCH --job-name cb_eval_rerun
#SBATCH --output /rds/user/wz369/hpc-work/PINN_dynamics/logs/cordblood/slurm/eval_rerun-%J.out
#SBATCH --error  /rds/user/wz369/hpc-work/PINN_dynamics/logs/cordblood/slurm/eval_rerun-%J.err

# Re-run eval for the 18 tier-1 cases that trained OK but eval-failed in job 29544952:
#   - prescient (PCA, DM) x 3 seeds = 6  (CUBLAS race -> now serialised via flock)
#   - otcfm    (PCA, DM) x 3 seeds = 6  (CLI arg mismatch -> fixed in dispatcher)
#   - sf2m     (PCA, DM) x 3 seeds = 6  (CLI arg mismatch -> fixed in dispatcher)
# pdp PCA (no checkpoint, val-empty bug) is NOT in this script; needs full retrain.

set -uo pipefail   # NOTE: no -e — prescient eval may fail per-seed and we want to keep going

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

cd /rds/user/wz369/hpc-work/PINN_dynamics

PY_PINN=/rds/user/wz369/hpc-work/LIBS/mamba/envs/PINN_env/bin/python
FLOW=/rds/user/wz369/hpc-work/containers/flow.sif
DATA=/rds/user/wz369/hpc-work/PINN_dynamics/data/cordblood_addpop.h5ad
F_OBS=data/CordBlood/F_obs_day17.csv
CLONE=data/CordBlood/clone_proportions_day17.csv

SEEDS=(42 7 1234)
EMBS=("X_pca_scaled" "DM_EigenVectors_scaled")

PRESCIENT_LOCK="${TMPDIR:-/tmp}/prescient_eval_${SLURM_JOB_ID}.lock"

run_one() {
    local method="$1" emb="$2" seed="$3"
    local outdir="logs/cordblood/${method}/${emb}_s${seed}"
    local log="${outdir}/eval_rerun.log"
    echo "[$(date)] ${method} ${emb} s${seed} -> ${outdir}"

    if [[ "$method" == "prescient" ]]; then
        local model_dir
        model_dir=$(find "${outdir}/train" -name "train.best.pt" -exec dirname {} \; | head -1)
        if [[ -z "$model_dir" ]]; then
            echo "  WARN: prescient train.best.pt missing under ${outdir}/train — skip" | tee -a "$log" >&2
            return
        fi
        # PRESCIENT W2 sim crashes with CUBLAS_STATUS_NOT_INITIALIZED on GPU in
        # this container (model loads fine but bulk sim trips it). Fall back to
        # --device cpu — slow but reliable. Fate eval already runs on CPU here.
        singularity exec "$FLOW" python scripts/prescient/03_evaluate.py \
            --model_dir "$model_dir" \
            --data_path "$DATA" \
            --obsm_key "$emb" --n_dims $([[ "$emb" == "DM_EigenVectors_scaled" ]] && echo 9 || echo 30) \
            --tp_col timepoint_tx_days --celltype_col def_lab \
            --device cpu \
            --F_obs_path "$F_OBS" --clone_proportions "$CLONE" \
            --output_dir "${outdir}/eval" >>"$log" 2>&1
    elif [[ "$method" == "otcfm" ]]; then
        singularity exec --nv "$FLOW" python scripts/otcfm/03_evaluate.py \
            --model_dir "$outdir" \
            --data_path "$DATA" --celltype_col def_lab \
            --F_obs_path "$F_OBS" --clone_proportions "$CLONE" >>"$log" 2>&1
    elif [[ "$method" == "sf2m" ]]; then
        singularity exec --nv "$FLOW" python scripts/sf2m/03_evaluate.py \
            --model_dir "$outdir" \
            --data_path "$DATA" --celltype_col def_lab \
            --F_obs_path "$F_OBS" --clone_proportions "$CLONE" >>"$log" 2>&1
    fi

    if [[ -f "${outdir}/eval/eval_combined.csv" ]] || [[ -f "${outdir}/eval_combined.csv" ]]; then
        echo "  PASS  ${method} ${emb} s${seed}"
    else
        echo "  FAIL  ${method} ${emb} s${seed} (no eval_combined.csv)" >&2
    fi
}

echo "[$(date)] Job $SLURM_JOB_ID on $(hostname)  GPU=${CUDA_VISIBLE_DEVICES:-unset}"

# Phase 1: otcfm + sf2m can run in parallel (already proven safe).
# Skip if eval_combined.csv already exists (idempotent re-runs).
pids=()
for method in otcfm sf2m; do
    for emb in "${EMBS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            if [[ -f "logs/cordblood/${method}/${emb}_s${seed}/eval_combined.csv" ]]; then
                echo "  SKIP (exists)  ${method} ${emb} s${seed}"
                continue
            fi
            (
                sleep $((RANDOM % 15))
                run_one "$method" "$emb" "$seed"
            ) &
            pids+=($!)
        done
    done
done

echo "[$(date)] Phase-1 PIDs (otcfm + sf2m): ${pids[*]}"
for pid in "${pids[@]}"; do
    wait "$pid" || echo "[$(date)] pid=$pid exited non-zero"
done
echo "[$(date)] Phase 1 done. Waiting 5s for GPU to clear."
sleep 5

# Phase 2: PRESCIENT runs strictly sequential to avoid CUBLAS init race.
# (parallel runs of prescient share a GPU process and reliably hit
#  CUBLAS_STATUS_NOT_INITIALIZED on any concurrent eval.)
for emb in "${EMBS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if [[ -f "logs/cordblood/prescient/${emb}_s${seed}/eval/eval_combined.csv" ]]; then
            echo "  SKIP (exists)  prescient ${emb} s${seed}"
            continue
        fi
        run_one "prescient" "$emb" "$seed"
    done
done

echo "[$(date)] All eval re-runs finished."
