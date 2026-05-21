#!/bin/bash
#SBATCH --job-name=pdp_c1c2_eval
#SBATCH --account=GOTTGENS-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/rds/user/wz369/hpc-work/pseudodynamics_plus/logs/synthetic_FP_5D_ablation/eval/c1c2_%j.out
#SBATCH --error=/rds/user/wz369/hpc-work/pseudodynamics_plus/logs/synthetic_FP_5D_ablation/eval/c1c2_%j.err

# C1 + C2 evaluation for the synthetic_FP_5D λ_growth sweep.
# C1 sweeps all (version, arm, seed) → per_seed.csv + summary.csv + sensitivity_curve.png.
# C2 runs the NN extrapolation diagnostic once per arm/seed (15 invocations).

set -u
PDP_DIR=/rds/user/wz369/hpc-work/pseudodynamics_plus
PY=/rds/user/wz369/hpc-work/LIBS/mamba/envs/PINN_env/bin/python
EVAL_DIR=${PDP_DIR}/logs/synthetic_FP_5D_ablation/eval
mkdir -p "${EVAL_DIR}/c2"

cd "${PDP_DIR}"
echo "[JOB] pdp_c1c2_eval job=${SLURM_JOB_ID:-local} node=$(hostname) start=$(date '+%F %T')"

# ─── C1: sweep all arms/seeds, write per_seed.csv + summary.csv + figure ────
if [ "${SKIP_C1:-0}" = "1" ]; then
    echo "[C1] SKIPPED (SKIP_C1=1)"
else
    echo "════════════════════════════════════════════════════════════════════"
    echo "[C1] running c1_eval.py (v2 only — synds v3 not run yet)"
    echo "════════════════════════════════════════════════════════════════════"
    t0=$(date +%s)
    ${PY} scripts/rebuttal_C1C2/c1_eval.py --version v2
    echo "[C1] elapsed=$(( $(date +%s) - t0 ))s"
fi

# ─── C2: NN extrapolation diagnostic, one run per (arm, seed) ───────────────
echo "════════════════════════════════════════════════════════════════════"
echo "[C2] running c2_extrapolation_diagnostic.py per (arm, seed) — 15 runs"
echo "════════════════════════════════════════════════════════════════════"

find_best_ckpt() {
    # Echo the lowest-val_loss .ckpt under a seed dir, or empty if none.
    local seed_dir="$1"
    local best_ckpt=""
    local best_loss="999999999"
    # Search every version_*/checkpoints under pde_params_tsense/lightning_logs
    while IFS= read -r ck; do
        [ -f "$ck" ] || continue
        local vl
        vl=$(basename "$ck" | grep -oP 'val_loss=\K-?[0-9.]+(?=\.ckpt)')
        [ -n "$vl" ] || continue
        if python3 -c "exit(0 if $vl < $best_loss else 1)" 2>/dev/null; then
            best_loss="$vl"
            best_ckpt="$ck"
        fi
    done < <(find "${seed_dir}/pde_params_tsense/lightning_logs" -name '*.ckpt' 2>/dev/null)
    echo "$best_ckpt"
}

for arm in lambdag_0 lambdag_1 lambdag_3 lambdag_10 lambdag_100; do
    for s in 0 1 2; do
        seed_dir="${PDP_DIR}/logs/synthetic_FP_5D_ablation/${arm}/seed_${s}"
        ckpt=$(find_best_ckpt "$seed_dir")
        if [ -z "$ckpt" ]; then
            echo "[C2 SKIP] ${arm}/seed_${s}  (no ckpt)"
            continue
        fi
        out_dir="${EVAL_DIR}/c2/${arm}_seed${s}"
        echo "─── [C2] ${arm}/seed_${s}  ckpt=$(basename "$ckpt")"
        t0=$(date +%s)
        ${PY} scripts/rebuttal_C1C2/c2_extrapolation_diagnostic.py \
            --ckpt "$ckpt" \
            --h5ad data/synthetic_FP_5D.h5ad \
            --out-dir "$out_dir"
        echo "[C2] ${arm}/seed_${s} rc=$? elapsed=$(( $(date +%s) - t0 ))s"
    done
done

echo "[JOB] pdp_c1c2_eval end=$(date '+%F %T')"
ls -la "${EVAL_DIR}/"
