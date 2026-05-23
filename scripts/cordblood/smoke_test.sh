#!/usr/bin/env bash
# smoke_test.sh — quick smoke run for cord-blood tier-1 plumbing.
#
# Runs one PdP task (PCA, seed 42, max_epochs=5) and one OT-CFM task
# (PCA, seed 42, niters=200) end-to-end to verify:
#   - dispatcher routes to the right Python env
#   - data loading from data/cordblood_addpop.h5ad works
#   - the well_col compat column allows the right test set
#   - eval scripts read F_obs_day17.csv / clone_proportions_day17.csv
#
# Each smoke run lives under logs/cordblood_smoke/<method>/.
# Set SMOKE_PDP=0 or SMOKE_OTCFM=0 to skip a method.
set -euo pipefail

PROJ=/rds/user/wz369/hpc-work/PINN_dynamics
cd "$PROJ"

DATA="$PROJ/data/cordblood_addpop.h5ad"
[[ -f "$DATA" ]] || { echo "Missing $DATA — run 01_preprocess.py first"; exit 2; }

SMOKE_DIR="logs/cordblood_smoke"
mkdir -p "$SMOKE_DIR"

PYTHON_PINN=/rds/user/wz369/hpc-work/LIBS/mamba/envs/PINN_env/bin/python
PYTHON_FLOW="singularity exec --nv /rds/user/wz369/hpc-work/containers/flow.sif python"
PDP_PLUS=/rds/user/wz369/hpc-work/pseudodynamics_plus

SMOKE_PDP=${SMOKE_PDP:-1}
SMOKE_OTCFM=${SMOKE_OTCFM:-1}

# Resolve dm_n_dims from the adata so the smoke run picks up the actual value
DM_N_DIMS=$($PYTHON_PINN -c "
import scanpy as sc
print(int(sc.read_h5ad('$DATA', backed='r').uns['dm_n_dims']))
")
echo "smoke: dm_n_dims = $DM_N_DIMS"

# ─── PdP smoke (PCA, seed=42, max_epochs=5) ──────────────────────────────────
if [[ "$SMOKE_PDP" == "1" ]]; then
    echo ""
    echo "===== SMOKE 1: pdp PCA seed=42 max_epochs=5 ====="
    PDP_OUT="$SMOKE_DIR/pdp_pca_s42"
    mkdir -p "$PDP_OUT"

    # Build a smoke config from the canonical PCA seed=42 config
    CANONICAL_CFG="logs/cordblood/pdp/X_pca_scaled_s42/V0_config.json"
    [[ -f "$CANONICAL_CFG" ]] || { echo "Missing canonical config: $CANONICAL_CFG (run _generate_configs.py)"; exit 2; }
    SMOKE_CFG="$PDP_OUT/V0_smoke_config.json"
    # Set save_dir / log_name to the smoke dir so we don't pollute the canonical config's outdir.
    $PYTHON_PINN - "$CANONICAL_CFG" "$SMOKE_CFG" "$PDP_OUT" <<'PY'
import json, sys
src, dst, outdir = sys.argv[1:]
with open(src) as fh: cfg = json.load(fh)
cfg["raw_args"]["log_name"] = "cordblood_smoke/pdp_pca_s42"
with open(dst, "w") as fh: json.dump(cfg, fh, indent=4)
print(f"Wrote smoke config: {dst}")
PY

    LOG="$SMOKE_DIR/pdp_pca_s42.log"
    echo "Running pdp smoke ... log → $LOG"
    set +e
    ( cd "$PDP_PLUS" && $PYTHON_PINN main_train.py \
        --config "$PROJ/$SMOKE_CFG" \
        -G 0 --seed 42 --max_epochs 5 ) 2>&1 | tee "$LOG"
    PDP_EXIT=$?
    set -e
    echo "pdp smoke exit: $PDP_EXIT"
    [[ "$PDP_EXIT" == "0" ]] && echo "✅ pdp smoke PASSED" || { echo "❌ pdp smoke FAILED"; exit "$PDP_EXIT"; }
fi

# ─── OT-CFM smoke (PCA, seed=42, niters=200) ─────────────────────────────────
if [[ "$SMOKE_OTCFM" == "1" ]]; then
    echo ""
    echo "===== SMOKE 2: otcfm PCA seed=42 niters=200 ====="
    OTCFM_OUT="$SMOKE_DIR/otcfm_pca_s42"
    mkdir -p "$OTCFM_OUT"
    LOG="$SMOKE_DIR/otcfm_pca_s42.log"

    echo "Running OT-CFM training ..."
    set +e
    $PYTHON_FLOW scripts/otcfm/02_train.py \
        --data_path "$DATA" \
        --obsm_key X_pca_scaled --n_dims 30 \
        --tp_col timepoint_tx_days \
        --save_dir "$OTCFM_OUT" \
        --niters 200 --batch_size 256 --sigma 0.1 --lr 1e-4 \
        --seed 42 --gpu 0 \
        --no_standardize 2>&1 | tee "$LOG"
    TRAIN_EXIT=$?
    set -e
    echo "otcfm train exit: $TRAIN_EXIT"
    if [[ "$TRAIN_EXIT" != "0" ]]; then
        echo "❌ otcfm train FAILED"; exit "$TRAIN_EXIT"
    fi

    echo "Running OT-CFM evaluation ..."
    set +e
    $PYTHON_FLOW scripts/otcfm/03_evaluate.py \
        --model_dir "$OTCFM_OUT" \
        --data_path "$DATA" \
        --F_obs_path data/CordBlood/F_obs_day17.csv \
        --clone_proportions data/CordBlood/clone_proportions_day17.csv \
        --celltype_col def_lab \
        --device "cuda:0" \
        --output_dir "$OTCFM_OUT" 2>&1 | tee -a "$LOG"
    EVAL_EXIT=$?
    set -e
    echo "otcfm eval exit: $EVAL_EXIT"
    [[ "$EVAL_EXIT" == "0" ]] && echo "✅ otcfm smoke PASSED" || { echo "❌ otcfm smoke FAILED"; exit "$EVAL_EXIT"; }
fi

echo ""
echo "===== SMOKE TEST SUMMARY ====="
ls -la "$SMOKE_DIR"
echo ""
echo "All smoke tests passed."
