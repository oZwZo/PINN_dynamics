#!/usr/bin/env bash
# smoke_pilot.sh — 1–2-epoch pilot for each tier-1 trainable method.
#
# Runs in-place on the current GPU node (no SLURM). Continues on failure so all
# methods get a chance to surface their bugs; reports a summary at the end.
#
# Methods (in order):
#   pdp     PCA  max_epochs=2  (smoke 0)
#   pdp     DM   max_epochs=2  (smoke 1; tests dm_n_dims=9 path)
#   otcfm   PCA  niters=200    (smoke 2)
#   sf2m    PCA  niters=200    (smoke 3)
#   prescient PCA pretrain=5 train=5 (smoke 4)
#
# Each method writes to logs/cordblood_smoke/<method>_<emb>_s42/ and a log file.
set -u  # NOT -e — we want to continue on per-method failures

PROJ=/rds/user/wz369/hpc-work/PINN_dynamics
cd "$PROJ"

DATA="$PROJ/data/cordblood_addpop.h5ad"
[[ -f "$DATA" ]] || { echo "Missing $DATA"; exit 2; }

PYTHON_PINN=/rds/user/wz369/hpc-work/LIBS/mamba/envs/PINN_env/bin/python
PYTHON_FLOW="singularity exec --nv /rds/user/wz369/hpc-work/containers/flow.sif python"
PDP_PLUS=/rds/user/wz369/hpc-work/pseudodynamics_plus

SMOKE_DIR="logs/cordblood_smoke"
mkdir -p "$SMOKE_DIR"

declare -a RESULTS=()

run_step() {
    local label="$1"; shift
    local logfile="$SMOKE_DIR/${label}.log"
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "[smoke] $label   log: $logfile"
    echo "═══════════════════════════════════════════════════════════════════"
    "$@" 2>&1 | tee "$logfile"
    local rc="${PIPESTATUS[0]}"
    if [[ "$rc" == "0" ]]; then
        echo "[smoke] $label  ✅ PASS"
        RESULTS+=("PASS  $label")
    else
        echo "[smoke] $label  ❌ FAIL (rc=$rc) — see $logfile"
        RESULTS+=("FAIL  $label  (rc=$rc)")
    fi
    return 0
}

run_pdp() {
    local embedding="$1"; local emb_name="$2"
    local OUT="$SMOKE_DIR/pdp_${emb_name}_s42"
    mkdir -p "$OUT"
    local CANONICAL="logs/cordblood/pdp/${embedding}_s42/V0_config.json"
    [[ -f "$CANONICAL" ]] || { echo "Missing canonical config $CANONICAL"; return 2; }
    local CFG="$OUT/V0_smoke_config.json"
    $PYTHON_PINN - "$CANONICAL" "$CFG" "$emb_name" <<'PY'
import json, sys
src, dst, emb = sys.argv[1:]
with open(src) as fh: cfg = json.load(fh)
cfg["raw_args"]["log_name"] = f"cordblood_smoke/pdp_{emb}_s42"
with open(dst, "w") as fh: json.dump(cfg, fh, indent=4)
PY
    ( cd "$PDP_PLUS" && $PYTHON_PINN main_train.py \
        --config "$PROJ/$CFG" -G 0 --seed 42 --max_epochs 2 )
}

run_otcfm() {
    local OUT="$SMOKE_DIR/otcfm_pca_s42"
    rm -rf "$OUT"; mkdir -p "$OUT"
    $PYTHON_FLOW scripts/otcfm/02_train.py \
        --data_path "$DATA" \
        --obsm_key X_pca_scaled --n_dims 30 \
        --tp_col timepoint_tx_days \
        --save_dir "$OUT" \
        --niters 200 --batch_size 256 --sigma 0.1 --lr 1e-4 \
        --seed 42 --gpu 0 \
        --no_standardize
}

run_sf2m() {
    local OUT="$SMOKE_DIR/sf2m_pca_s42"
    rm -rf "$OUT"; mkdir -p "$OUT"
    $PYTHON_FLOW scripts/sf2m/02_train.py \
        --data_path "$DATA" \
        --obsm_key X_pca_scaled --n_dims 30 \
        --tp_col timepoint_tx_days \
        --save_dir "$OUT" \
        --niters 200 --batch_size 256 --sigma 0.05 --lr 1e-4 \
        --seed 42 --gpu 0 \
        --no_standardize
}

run_prescient() {
    local OUT="$SMOKE_DIR/prescient_pca_s42"
    rm -rf "$OUT"; mkdir -p "$OUT"
    $PYTHON_FLOW scripts/prescient/01_prepare_data.py \
        --data_path "$DATA" \
        --output_dir "$SMOKE_DIR" --run_name prescient_pca_s42 \
        --obsm_key X_pca_scaled --n_dims 30 \
        --tp_col timepoint_tx_days \
        --well_col split --test_values test \
        --celltype_col def_lab
    local rc=$?
    [[ $rc != 0 ]] && return $rc
    $PYTHON_FLOW scripts/prescient/02_train.py \
        --data_dir "$OUT" --exp_name pca_smoke --weight_name smoke \
        --pretrain_epochs 5 --train_epochs 5 --save 5 \
        --seed 42 --gpu 0
}

run_step "pdp_pca_s42_e2"   run_pdp X_pca_scaled pca
run_step "pdp_dm_s42_e2"    run_pdp DM_EigenVectors_scaled dm
run_step "otcfm_pca_s42_n200" run_otcfm
run_step "sf2m_pca_s42_n200"  run_sf2m
run_step "prescient_pca_s42_e5" run_prescient

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "                       SMOKE PILOT SUMMARY"
echo "═══════════════════════════════════════════════════════════════════"
for r in "${RESULTS[@]}"; do echo "  $r"; done
