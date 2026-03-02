#!/usr/bin/env bash
# train_klein.sh — Train all Klein lambda-ablation configurations (18 total)
#
# Usage:
#   bash scripts/train_klein.sh [GPU_ID]
#
# Training order (rationale):
#   1. DM baseline          — 10-dim, fast training; establishes reference
#   2. DM λD sweep          — diffusion penalty: [0.1, 10]
#   3. DM λv sweep          — velocity alignment: [0.01, 0.1, 10]
#   4. DM λg sweep          — growth weight:     [0, 0.1, 1]
#   5–8. Same for PC (30-dim, slower — runs after all DM results are available)
#
# Features:
#   - Skips configs that already have a checkpoint
#   - Logs each run's stdout/stderr to logs/<name>_train.log
#   - Reports per-run and total elapsed time
#
set -euo pipefail

PYTHON=/rds/user/wz369/hpc-work/LIBS/mamba/envs/PINN_env/bin/python
GPU=${1:-0}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"
echo "Working directory: $PROJECT_DIR"
echo "Python: $PYTHON"
echo "GPU: $GPU"

TOTAL_START=$(date +%s)
N_TRAINED=0
N_SKIPPED=0

# ── helper ──────────────────────────────────────────────────────────────────
run_config() {
    local cfg="$1"
    local cfg_dir name ckpt_dir

    cfg_dir=$(dirname "$cfg")
    name=$(basename "$cfg_dir")
    ckpt_dir="${cfg_dir}/pde_params_tsense/lightning_logs"

    # Skip if any checkpoint already exists
    if find "$ckpt_dir" -name "*.ckpt" 2>/dev/null | grep -q .; then
        echo "[SKIP]  $name — checkpoint already exists"
        N_SKIPPED=$((N_SKIPPED + 1))
        return 0
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[TRAIN] $name"
    echo "        $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local t0 t1
    t0=$(date +%s)

    $PYTHON main_train.py --config "$cfg" -G "$GPU" 2>&1 | tee "logs/${name}_train.log"

    t1=$(date +%s)
    echo "[DONE]  $name — elapsed: $((t1 - t0))s"
    N_TRAINED=$((N_TRAINED + 1))
}

# ── DM (10-dim) — runs first: smaller model, faster feedback ─────────────────

# 1. Baseline — central reference point for all comparisons
run_config logs/klein_DM_10_lD1_lv1_lgNone/V0_base.json


# 2. λD sweep — hold λv=1, λg=None; vary D_penalty
run_config logs/klein_DM_10_lD0.1_lv1_lgNone/V0_base.json
run_config logs/klein_DM_10_lD10_lv1_lgNone/V0_base.json

# 3. λv sweep — hold λD=1, λg=None; vary deltax_weight
run_config logs/klein_DM_10_lD1_lv0.01_lgNone/V0_base.json
run_config logs/klein_DM_10_lD1_lv0.1_lgNone/V0_base.json
run_config logs/klein_DM_10_lD1_lv10_lgNone/V0_base.json

# 4. λg sweep — hold λD=1, λv=1; vary growth_weight
run_config logs/klein_DM_10_lD1_lv1_lg0/V0_base.json
run_config logs/klein_DM_10_lD1_lv1_lg0.1/V0_base.json
run_config logs/klein_DM_10_lD1_lv1_lg1/V0_base.json

# ── PC (30-dim) — same order as DM ───────────────────────────────────────────

# 5. Baseline
run_config logs/klein_PC_30_lD1_lv1_lgNone/V0_base.json

# 6. λD sweep
run_config logs/klein_PC_30_lD0.1_lv1_lgNone/V0_base.json
run_config logs/klein_PC_30_lD10_lv1_lgNone/V0_base.json

# 7. λv sweep
run_config logs/klein_PC_30_lD1_lv0.01_lgNone/V0_base.json
run_config logs/klein_PC_30_lD1_lv0.1_lgNone/V0_base.json
run_config logs/klein_PC_30_lD1_lv10_lgNone/V0_base.json

# 8. λg sweep
run_config logs/klein_PC_30_lD1_lv1_lg0/V0_base.json
run_config logs/klein_PC_30_lD1_lv1_lg0.1/V0_base.json
run_config logs/klein_PC_30_lD1_lv1_lg1/V0_base.json

# ── summary ──────────────────────────────────────────────────────────────────
TOTAL_END=$(date +%s)
TOTAL_SEC=$((TOTAL_END - TOTAL_START))
TOTAL_MIN=$((TOTAL_SEC / 60))

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  Trained : $N_TRAINED   Skipped : $N_SKIPPED"
echo "  Total   : ${TOTAL_MIN}m $((TOTAL_SEC % 60))s"
echo "════════════════════════════════════════════════════════════════════════"
