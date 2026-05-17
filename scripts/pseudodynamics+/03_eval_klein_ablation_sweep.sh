#!/bin/bash
# Sequential W2 evaluation sweep over klein_OT + klein_nonOT ablation arms.
#
# For each (dataset, arm, seed) under
#   logs/{klein_OT,klein_nonOT}_ablation/<arm>/seed_<n>/pde_params_tsense/
# pick the V<jobid>_config.json whose referenced checkpoint has the highest
# epoch number, then call 03_eval_pdp_w2.py --skip_per_clone with that config.
#
# Outputs:
#   results/pseudodynamics+/<dataset>_ablation__<arm>__seed_<n>/
#       t<T_END>_n<NOISE>_<SIM_FN>_w2_eval.csv
#
# Idempotent: a (arm, seed) is skipped if its output CSV already exists.
# On failure, the script logs the error and continues to the next eval.
#
# Usage:
#   bash scripts/pseudodynamics+/03_eval_klein_ablation_sweep.sh
#   # Knobs (set via env vars):
#   #   DATASETS="klein_OT klein_nonOT"   # space-separated
#   #   SIM_FN=ode                        # ode | sde | sb
#   #   T_END=2.5                         # --t_end_norm
#   #   NOISE=1.0                         # --noise_scale
#   #   DEVICE=cuda:0
#   #   PY=/local/scratch/wz369/PINN_env/bin/python
#   #   FORCE=0                           # 1 to overwrite existing CSVs

set -u
PDP_DIR=/rds/user/wz369/hpc-work/pseudodynamics_plus
PY=${PY:-/local/scratch/wz369/PINN_env/bin/python}
EVAL_PY="${PDP_DIR}/scripts/pseudodynamics+/03_eval_pdp_w2.py"

DATASETS=${DATASETS:-"klein_OT klein_nonOT"}
SIM_FN=${SIM_FN:-sde}
T_END=${T_END:-3.0}
NOISE=${NOISE:-1.5}
DEVICE=${DEVICE:-cuda:0}
FORCE=${FORCE:-0}

LOG_DIR="${PDP_DIR}/logs/eval_klein/ablation_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
SUMMARY="${LOG_DIR}/_summary.tsv"
TODO_TSV="${LOG_DIR}/_todo.tsv"

echo "[SWEEP] start=$(date '+%F %T')  node=$(hostname)  device=${DEVICE}"
echo "[SWEEP] PY=${PY}"
echo "[SWEEP] DATASETS=${DATASETS}  SIM_FN=${SIM_FN}  T_END=${T_END}  NOISE=${NOISE}"
echo "[SWEEP] log_dir=${LOG_DIR}"
echo

# ── Step 1: build the TODO TSV (arm × seed → cfg, ckpt, epoch, out) ────────
${PY} - "${PDP_DIR}" "${T_END}" "${NOISE}" "${SIM_FN}" "${TODO_TSV}" ${DATASETS} <<'PYEOF'
import json, re, sys
from pathlib import Path

PDP = Path(sys.argv[1])
T_END = sys.argv[2]
NOISE = sys.argv[3]
SIM_FN = sys.argv[4]
TODO = Path(sys.argv[5])
DATASETS = sys.argv[6:]
RES_ROOT = PDP / "results" / "pseudodynamics+"
EXPECTED = f"t{T_END}_n{NOISE}_{SIM_FN}_w2_eval.csv"
CKPT_RE = re.compile(r"epoch=(\d+)-val_loss=(-?[0-9.]+)\.ckpt")

def best_ckpt(version_dir: Path):
    cd = version_dir / "checkpoints"
    if not cd.is_dir():
        return None, -1
    best, best_ep = None, -1
    for c in cd.glob("*.ckpt"):
        m = CKPT_RE.match(c.name)
        if not m:
            continue
        ep = int(m.group(1))
        if ep > best_ep:
            best_ep = ep
            best = c
    return best, best_ep

def pick_config(seed_dir: Path):
    pp = seed_dir / "pde_params_tsense"
    if not pp.is_dir():
        return None, None, -1
    best_cfg, best_ckpt_p, best_ep = None, None, -1
    for cfg in pp.glob("V*_config.json"):
        try:
            d = json.loads(cfg.read_text())
        except Exception:
            continue
        ckpt_dir = d.get("experiment_config", {}).get("checkpoint_dir")
        if not ckpt_dir:
            continue
        c, ep = best_ckpt(Path(ckpt_dir))
        if ep > best_ep:
            best_ep, best_cfg, best_ckpt_p = ep, cfg, c
    return best_cfg, best_ckpt_p, best_ep

rows = []
for ds in DATASETS:
    ds_root = PDP / "logs" / f"{ds}_ablation"
    if not ds_root.is_dir():
        print(f"[WARN] missing dataset root: {ds_root}", file=sys.stderr)
        continue
    for arm_dir in sorted(ds_root.glob("*")):
        if not arm_dir.is_dir():
            continue
        arm = arm_dir.name
        for s in [0, 1, 2]:
            seed_dir = arm_dir / f"seed_{s}"
            if not seed_dir.is_dir():
                continue
            cfg, ckpt, ep = pick_config(seed_dir)
            name = f"{ds}_ablation__{arm}__seed_{s}"
            out = RES_ROOT / name / EXPECTED
            if cfg is None or ckpt is None:
                rows.append((ds, arm, s, "NOCKPT", "", -1, str(out)))
            else:
                rows.append((ds, arm, s, str(cfg), str(ckpt), ep, str(out)))

with open(TODO, "w") as f:
    f.write("dataset\tarm\tseed\tconfig\tckpt\tepoch\toutput\n")
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")
print(f"Wrote {TODO} with {len(rows)} (dataset, arm, seed) entries")
PYEOF

# ── Step 2: loop through the TODO TSV, run each eval sequentially ──────────
printf "status\tdataset\tarm\tseed\tepoch\telapsed_s\toutput\n" > "${SUMMARY}"

n_total=0; n_done=0; n_skip=0; n_fail=0; n_nockpt=0
while IFS=$'\t' read -r dataset arm seed config ckpt epoch output; do
    [ "${dataset}" = "dataset" ] && continue   # skip header
    n_total=$((n_total + 1))
    tag="${dataset}/${arm}/seed_${seed}"

    if [ "${config}" = "NOCKPT" ]; then
        echo "[${n_total}] SKIP (no ckpt) ${tag}"
        printf "NOCKPT\t%s\t%s\t%s\t-1\t0\t%s\n" "${dataset}" "${arm}" "${seed}" "${output}" >> "${SUMMARY}"
        n_nockpt=$((n_nockpt + 1))
        continue
    fi

    if [ -f "${output}" ] && [ "${FORCE}" != "1" ]; then
        echo "[${n_total}] DONE (cached) ${tag}  ep=${epoch}"
        printf "CACHED\t%s\t%s\t%s\t%s\t0\t%s\n" "${dataset}" "${arm}" "${seed}" "${epoch}" "${output}" >> "${SUMMARY}"
        n_skip=$((n_skip + 1))
        continue
    fi

    log_file="${LOG_DIR}/${dataset}__${arm}__seed_${seed}.log"
    echo "[${n_total}] RUN  ${tag}  ep=${epoch}  cfg=$(basename "${config}")"
    t0=$(date +%s)
    if cd "${PDP_DIR}" && ${PY} "${EVAL_PY}" \
            --sim_fn "${SIM_FN}" \
            --t_end_norm "${T_END}" \
            --noise_scale "${NOISE}" \
            --config_path "${config}" \
            --device "${DEVICE}" \
            --skip_per_clone \
            > "${log_file}" 2>&1; then
        elapsed=$(( $(date +%s) - t0 ))
        echo "       ok  elapsed=${elapsed}s  → ${output}"
        printf "OK\t%s\t%s\t%s\t%s\t%d\t%s\n" "${dataset}" "${arm}" "${seed}" "${epoch}" "${elapsed}" "${output}" >> "${SUMMARY}"
        n_done=$((n_done + 1))
    else
        elapsed=$(( $(date +%s) - t0 ))
        echo "       FAIL elapsed=${elapsed}s  log=${log_file}"
        printf "FAIL\t%s\t%s\t%s\t%s\t%d\t%s\n" "${dataset}" "${arm}" "${seed}" "${epoch}" "${elapsed}" "${log_file}" >> "${SUMMARY}"
        n_fail=$((n_fail + 1))
    fi
done < "${TODO_TSV}"

echo
echo "[SWEEP] end=$(date '+%F %T')"
echo "[SWEEP] total=${n_total}  ok=${n_done}  cached=${n_skip}  nockpt=${n_nockpt}  fail=${n_fail}"
echo "[SWEEP] summary → ${SUMMARY}"
echo "[SWEEP] per-eval stdout/stderr → ${LOG_DIR}/<dataset>__<arm>__seed_<n>.log"
