#!/usr/bin/env bash
# smoke_tier2.sh — quick smoke for MIOFlow / TIGON / TrajectoryNet on cord blood.
#
# Uses the env-var overrides in dispatch_task.sh to reduce iters/epochs.
# Each method is run with a single seed (42) and embedding (PCA scaled) so the
# smoke completes in minutes. After smoke passes, the full tier-2 SLURM array
# can be submitted.
set -uo pipefail

PROJ=/rds/user/wz369/hpc-work/PINN_dynamics
cd "$PROJ"

DATA="$PROJ/data/cordblood_addpop.h5ad"
[[ -f "$DATA" ]] || { echo "Missing $DATA"; exit 2; }

SMOKE_DIR="logs/cordblood_smoke"
mkdir -p "$SMOKE_DIR"

# Aggressively-reduced iters/epochs for the smoke
export MIOFLOW_EPOCHS=2
export TIGON_NITERS=200
export TRAJECTORYNET_NITERS=200

declare -a RESULTS=()

run_step() {
    local label="$1"; shift
    local logfile="$SMOKE_DIR/${label}.log"
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "[tier2-smoke] $label   log: $logfile"
    echo "═══════════════════════════════════════════════════════════════════"
    "$@" 2>&1 | tee "$logfile"
    local rc="${PIPESTATUS[0]}"
    if [[ "$rc" == "0" ]]; then
        echo "[tier2-smoke] $label  ✅ PASS"
        RESULTS+=("PASS  $label")
    else
        echo "[tier2-smoke] $label  ❌ FAIL (rc=$rc)"
        RESULTS+=("FAIL  $label  (rc=$rc)")
    fi
    return 0
}

run_mioflow() {
    local OUT="$SMOKE_DIR/mioflow_pca_s42"; rm -rf "$OUT"; mkdir -p "$OUT"
    scripts/cordblood/dispatch_task.sh mioflow X_pca_scaled 42 "$OUT" "$DATA"
}
run_tigon() {
    local OUT="$SMOKE_DIR/tigon_pca_s42"; rm -rf "$OUT"; mkdir -p "$OUT"
    scripts/cordblood/dispatch_task.sh tigon X_pca_scaled 42 "$OUT" "$DATA"
}
run_trajectorynet() {
    local OUT="$SMOKE_DIR/trajectorynet_pca_s42"; rm -rf "$OUT"; mkdir -p "$OUT"
    scripts/cordblood/dispatch_task.sh trajectorynet X_pca_scaled 42 "$OUT" "$DATA"
}

run_step "tier2_mioflow"       run_mioflow
run_step "tier2_tigon"         run_tigon
run_step "tier2_trajectorynet" run_trajectorynet

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "             TIER-2 SMOKE SUMMARY"
echo "═══════════════════════════════════════════════════════════════════"
for r in "${RESULTS[@]}"; do echo "  $r"; done
