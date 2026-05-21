#!/bin/bash
# Evaluate 16 scDiffEq checkpoints on Klein sequentially on the current node.
#   - 12 seeded runs: 3 seeds × {pca30, dm10} × {plain_ode, plain_sde}
#   - 4 fp_vr runs (pca30 only): versions 0..3
#
# Each run emits into {parent_dir}/fate_prediction_metrics/{ckpt_name}/:
#   eval_combined.csv, F_hat_{mode}.csv, w2_per_clone_{mode}.csv
#
# Runs the scDiffEq project's uv env (scdiffeq + torchcfm installed there).
# Stops on first error so issues surface immediately.

set -e
set -o pipefail

UV_PROJECT=/rds/user/wz369/hpc-work/scDiffEq
EVAL=/home/wz369/rds/hpc-work/PINN_dynamics/scripts/scdiffeq/klein_fate_eval_scdiffeq.py
ROOT=/rds/user/wz369/hpc-work/scDiffEq/results

# eval hyperparameters (matches the original sbatch + sf2m defaults)
N=200        # SDE trajectories / cell for fate accuracy
K=15         # kNN neighbours
SEED=0       # split reconstruction (constant → same F_obs/x_ref/W2 cells everywhere)
NW2=10       # SDE trajectories / cell for W2

cd "$UV_PROJECT"

run_one () {
    local CFG=$1
    local CKPT=$2
    local NAME=$3
    local OUT=$4

    if [[ ! -f "$CKPT" ]]; then
        echo "!! MISSING: $CKPT" >&2
        return 1
    fi

    echo "============================================================"
    echo "=== $(date -Iseconds)  $CFG  $NAME"
    echo "    ckpt : $CKPT"
    echo "    out  : $OUT/fate_prediction_metrics/$NAME/"
    echo "============================================================"

    uv run python "$EVAL" \
        --config     "$CFG" \
        --ckpt_path  "$CKPT" \
        --ckpt_name  "$NAME" \
        --output_dir "$OUT" \
        --N          "$N" \
        --k          "$K" \
        --n_sims_w2  "$NW2" \
        --seed       "$SEED" \
        --device     cuda:0
}

# ── seeded sweep: 12 runs ──
for S in 0 1 2; do
    for C in pca30 dm10; do
        for V in plain_ode plain_sde; do
            if [[ "$V" == "plain_ode" ]]; then
                CLASS=LightningODE
            else
                CLASS=LightningSDE
            fi
            CKPT="${ROOT}/seeds/seed_${S}/${C}/${V}/${CLASS}/version_0/checkpoints/last.ckpt"
            OUT="${ROOT}/seeds/seed_${S}/${C}/${V}"
            run_one "$C" "$CKPT" "last" "$OUT"
        done
    done
done

# ── fp_vr sweep: 4 runs (pca30 only) ──
FP_ROOT="${ROOT}/pca30/LightningSDE-FixedPotential-RegularizedVelocityRatio"
for V in 0 1 2 3; do
    CKPT="${FP_ROOT}/version_${V}/checkpoints/last.ckpt"
    run_one "pca30" "$CKPT" "version_${V}.last" "$FP_ROOT"
done

echo
echo "Done."
