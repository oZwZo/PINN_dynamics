#!/usr/bin/env bash
# smoke_eval.sh — evaluate the smoke-trained pilot checkpoints.
#
# Methods covered (3, those that produced a checkpoint):
#   pdp     DM   → scripts/pseudodynamics+/03_evaluate.py
#   otcfm   PCA  → scripts/otcfm/03_evaluate.py
#   sf2m    PCA  → scripts/sf2m/03_evaluate.py
#
# Skipped:
#   pdp     PCA  → no checkpoint saved (smoke had no val data; ModelCheckpoint skipped)
#   prescient PCA → smoke crashed with geomloss NaN before producing a checkpoint
set -u

PROJ=/rds/user/wz369/hpc-work/PINN_dynamics
cd "$PROJ"

DATA="$PROJ/data/cordblood_addpop.h5ad"
F_OBS="data/CordBlood/F_obs_day17.csv"
CLONE_PROP="data/CordBlood/clone_proportions_day17.csv"

PYTHON_PINN=/rds/user/wz369/hpc-work/LIBS/mamba/envs/PINN_env/bin/python
PYTHON_FLOW="singularity exec --nv /rds/user/wz369/hpc-work/containers/flow.sif python"

SMOKE_DIR="logs/cordblood_smoke"
declare -a RESULTS=()

run_step() {
    local label="$1"; shift
    local logfile="$SMOKE_DIR/${label}.log"
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "[eval-smoke] $label"
    echo "  log: $logfile"
    echo "═══════════════════════════════════════════════════════════════════"
    "$@" 2>&1 | tee "$logfile"
    local rc="${PIPESTATUS[0]}"
    if [[ "$rc" == "0" ]]; then
        echo "[eval-smoke] $label  ✅ PASS"
        RESULTS+=("PASS  $label")
    else
        echo "[eval-smoke] $label  ❌ FAIL (rc=$rc)"
        RESULTS+=("FAIL  $label  (rc=$rc)")
    fi
    return 0
}

# ─── OT-CFM eval ──────────────────────────────────────────────────────────
eval_otcfm() {
    $PYTHON_FLOW scripts/otcfm/03_evaluate.py \
        --model_dir "$SMOKE_DIR/otcfm_pca_s42" \
        --data_path "$DATA" \
        --F_obs_path "$F_OBS" \
        --clone_proportions "$CLONE_PROP" \
        --celltype_col def_lab \
        --device cuda:0 \
        --n_sims_fate 10 --n_sims_w2 2 \
        --output_dir "$SMOKE_DIR/otcfm_pca_s42"
}

# ─── SF2M eval ────────────────────────────────────────────────────────────
eval_sf2m() {
    $PYTHON_FLOW scripts/sf2m/03_evaluate.py \
        --model_dir "$SMOKE_DIR/sf2m_pca_s42" \
        --data_path "$DATA" \
        --F_obs_path "$F_OBS" \
        --clone_proportions "$CLONE_PROP" \
        --celltype_col def_lab \
        --device cuda:0 \
        --n_sims_fate 10 --n_sims_w2 2 \
        --output_dir "$SMOKE_DIR/sf2m_pca_s42"
}

# ─── PdP DM eval ──────────────────────────────────────────────────────────
# pdp's 03_evaluate.py expects --config_path pointing to the TRAINING-time
# V0_config.json (with dataset_config/experiment_config sections populated),
# which is saved under <save_dir>/pde_params_tsense/V0_config.json by the
# pseudodynamics_plus trainer. Not the smoke INPUT config (which only has raw_args).
eval_pdp_dm() {
    local CFG=/rds/user/wz369/hpc-work/pseudodynamics_plus/logs/cordblood_smoke/pdp_dm_s42/pde_params_tsense/V0_config.json
    [[ -f "$CFG" ]] || { echo "Missing trained pdp config: $CFG"; return 2; }
    # Run from PINN_dynamics root so data/ paths resolve
    $PYTHON_PINN scripts/pseudodynamics+/03_evaluate.py \
        --config_path "$CFG" \
        --data_path "$DATA" \
        --F_obs_path "$F_OBS" \
        --clone_proportions "$CLONE_PROP" \
        --celltype_col def_lab \
        --n_sims_fate 5 --n_sims_w2 2 \
        --output_dir "$SMOKE_DIR/pdp_dm_s42/eval"
}

run_step "otcfm_eval"  eval_otcfm
run_step "sf2m_eval"   eval_sf2m
run_step "pdp_dm_eval" eval_pdp_dm

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "                  EVAL-SMOKE SUMMARY"
echo "═══════════════════════════════════════════════════════════════════"
for r in "${RESULTS[@]}"; do echo "  $r"; done
