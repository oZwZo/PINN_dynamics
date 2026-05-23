# Cord-Blood Benchmark: Analyser Report
**Generated:** 2026-05-23  
**Analyst:** automated result-analyser agent

---

## 1. Overview Table

| Method     | Embedding              | Seeds evaluated / 3 | Best W2_scaled (mode) | Status         |
|------------|------------------------|---------------------|-----------------------|----------------|
| otcfm      | DM_EigenVectors_scaled | 1/3                 | 1.90 (ode)            | PARTIAL        |
| prescient  | DM_EigenVectors_scaled | 1/3                 | 3.59 (sde)            | PARTIAL        |
| scdiffeq   | DM_EigenVectors_scaled | 1/3 (smoke)         | 6.51 (sde)            | SMOKE_ONLY     |
| pdp        | DM_EigenVectors_scaled | 3/3                 | 11.74 (ode, mean)     | COMPLETE       |
| pdp        | X_pca_scaled           | 1/3                 | 5.81 (ode)            | PARTIAL        |
| otcfm      | X_pca_scaled           | 0/3                 | —                     | PENDING        |
| sf2m       | DM_EigenVectors_scaled | 0/3                 | —                     | PENDING        |
| sf2m       | X_pca_scaled           | 0/3                 | —                     | PENDING        |
| prescient  | X_pca_scaled           | 0/3                 | —                     | PENDING        |
| deepruot   | DM_EigenVectors_scaled | 0/3                 | —                     | PENDING        |
| scDiffEq   | DM_EigenVectors_scaled | 0/3 (tier-2)        | —                     | PENDING        |

---

## 2. Per-Method Diagnosis

### pdp (PseudoDynamics+)
**DM — 3/3 seeds complete.** All three seeds (42, 7, 1234) have full eval across ode/sde/sb modes. Metrics are consistent across seeds (low variance): ode W2_scaled = 11.34 / 12.08 / 11.81, mean ≈ 11.74 (std 0.37). Pearson r is near-zero or slightly negative for ode/sde (range −0.07 to +0.12), suggesting fate predictions are barely above random for these modes. The sb mode shows slightly better Pearson r (up to 0.20 for s1234) but higher W2_scaled (~15.4–16.0). Accuracy is uniformly low (~7–11%). The low w2_raw (~0.08) for DM versus the high w2_scaled (~11–16) indicates the scaling denominator may be compressing differences — this should be verified.

**PCA — 1/3 seeds partially evaluated.** Only s7 (ode) has a result (W2_scaled=5.81, but Pearson r=−0.18 — anomalous). Seeds 42 and 1234 have no checkpoint due to the known val-empty bug; PCA retrain submitted as job 29570505. DM seeds 42/7 also lack V0_config.json (the training pipeline in the outer slurm logged "ERROR: pdp trained config not found" despite training completing to epoch 399), meaning the eval used checkpoints from a different run (version_29546281 for DM, version_29546236 for PCA s7). The DM eval results are valid since they have proper eval_combined.csv files.

### prescient
**DM — 1/3 seeds evaluated.** Only s42 has eval_combined.csv (W2_scaled=3.59, Pearson r=0.569, sde mode). Training completed for all 6 runs (done.log present for DM s42/7/1234 and PCA s42/7/1234 — all finished by 06:58 on 2026-05-23). However, the eval job (29570504) crashed for s7, s1234, and all PCA seeds with `RuntimeError: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED`. This is a GPU resource allocation issue during inference (not a model problem). The s42 DM eval must have run under a prior job. Re-runs are PENDING. Training quality looks good: val_loss converged from ~5.6 down to ~0.96 at epoch 2500.

### otcfm (OT-CFM)
**DM s42 — 1/6 cells evaluated.** Only DM s42 has eval_combined.csv (W2_scaled=1.90, Pearson r=0.737 — the best result in the benchmark so far). Training for DM s42 completed with best val_loss=0.2234 at iter 11000, early-stopping triggered at iter 13000. Training checkpoints are confirmed present (ckpt.pt) for all 6 cells. All other 5 cells (DM s7, DM s1234, PCA s42/7/1234) failed eval with identical error: `03_evaluate.py: error: unrecognized arguments: --obsm_key ... --n_dims ... --tp_col timepoint_tx_days`. This is a CLI argument version mismatch in the evaluate script — the tier-1 eval resubmission (job 29570504) should fix this.

### sf2m (Schrödinger Flow Matching)
**0/6 cells evaluated.** Training checkpoints confirmed present for all 6 cells (ckpt.pt, config.json, scaler.npz). Eval failed for all with the same CLI argument error as otcfm (`--obsm_key`, `--n_dims`, `--tp_col` unrecognized). This is the same bug as otcfm. No model-level anomalies expected — training ran to completion.

### scdiffeq (smoke run)
**1 smoke cell evaluated.** DM 9-dim, fp_vr config, s42: W2_scaled=6.51, Pearson r=0.602, sde mode. Reasonable performance as a smoke test. Full tier-2 jobs (29570508/29570509) are PENDING. Note this is a `scdiffeq_smoke` subdirectory, not the canonical `scdiffeq/DM_EigenVectors_scaled_s42` path.

### deepruot (smoke)
**0 cells with eval_combined.csv.** Smoke directory contains only `F_hat_ode.csv` (predictions exist but eval pipeline did not complete). Jobs 29570506/29570507 are PENDING.

---

## 3. Failed Runs — Summary and Remediation

| Run                             | Failure Mode                                   | Proposed Fix                                                    |
|---------------------------------|------------------------------------------------|-----------------------------------------------------------------|
| otcfm — all except DM s42 eval  | `--obsm_key`/`--n_dims`/`--tp_col` unrecognized by 03_evaluate.py | Update 03_evaluate.py argparser to accept these flags (or remove from call); rerun eval job 29570504 once fixed |
| sf2m — all eval                  | Same CLI mismatch as otcfm                     | Same fix as otcfm                                               |
| prescient — DM s7, s1234, all PCA evals | CUDA CUBLAS_STATUS_NOT_INITIALIZED at inference | Request dedicated GPU node; ensure no other CUDA processes on same device; re-submit eval with `CUDA_VISIBLE_DEVICES` guard |
| pdp PCA — s42, s1234            | val-empty bug → no best checkpoint            | Retrain already submitted (job 29570505); verify val split logic |
| pdp — all DM/PCA s42/s7 outer config | V0_config.json not found (wrong path)          | Verify `model_dir` path in eval script matches actual checkpoint location (version_29546281 vs expected) |
| deepruot — eval not written      | F_hat_ode.csv present but eval_combined.csv missing | Run fate_eval_pipeline on existing F_hat; jobs 29570506-7 pending |

---

## 4. Anomaly Flags

1. **pdp DM — Pearson r near-zero or negative across all 3 seeds and ode/sde modes.** Mean Pearson r for ode = +0.023 (range −0.066 to +0.117), for sde = +0.011. Fate prediction is essentially at random. Possible causes: DM embedding has 9 dims which may be insufficient for the cord-blood dataset's complexity; clone-proportion matching may require calibration. The sb mode shows marginally better r (up to 0.20) suggesting stochastic boundary conditions help.

2. **pdp X_pca_scaled s7 — Pearson r = −0.182 and w2_raw = 28.93 (anomalously high).** The w2_raw for DM runs is ~0.08 but PCA s7 reports 28.93. This likely reflects a unit/scale difference (PCA space has larger absolute magnitudes). The w2_scaled = 5.81 is more reasonable if the scaling is computed correctly. Still, negative Pearson r warrants attention.

3. **otcfm/sf2m eval crash — all 11 cells failed with same CLI error.** This is a systemic bug (argument name mismatch between the dispatch script and the evaluate script). The root cause is that the tier-1 `03_evaluate.py` for otcfm/sf2m does not declare `--obsm_key`, `--n_dims`, or `--tp_col` as valid arguments. This blocks the entire otcfm and sf2m leaderboard columns.

---

## 5. Leaderboard Ranking (current available results)

### DM_EigenVectors_scaled — ranked by mean W2_scaled (lower = better)

| Rank | Method    | Seeds | W2_scaled (mean) | Pearson r (mean) | Accuracy (mean) | Sim mode |
|------|-----------|-------|------------------|------------------|-----------------|----------|
| 1    | otcfm     | 1     | 1.90             | 0.737            | 0.359           | ode      |
| 2    | prescient | 1     | 3.59             | 0.569            | 0.248           | sde      |
| 3    | scdiffeq  | 1*    | 6.51             | 0.602            | 0.325           | sde      |
| 4    | pdp       | 3     | 11.74            | 0.023            | 0.077           | ode      |

*smoke run only

### X_pca_scaled — ranked by mean W2_scaled (lower = better)

| Rank | Method | Seeds | W2_scaled | Pearson r | Accuracy | Sim mode |
|------|--------|-------|-----------|-----------|----------|----------|
| 1    | pdp    | 1     | 5.81      | −0.182    | 0.111    | ode      |

All other PCA results are PENDING.

---

## 6. Klein Baseline Comparison

No Klein leaderboard CSV found at `results/klein/leaderboard.csv`. Available Klein eval CSVs are in `logs/TIGON/`, `logs/MIOFlow/`, `logs/TrajectoryNet/` — these use different method names than the tier-1 benchmark. Extracted spot-check values:

| Method (Klein) | Embedding | W2_scaled | Pearson r |
|----------------|-----------|-----------|-----------|
| TIGON          | DM        | 2.84      | 0.874     |
| TIGON          | PCA       | 4.25      | 0.981     |
| TrajectoryNet  | DM10      | 3.01      | 0.665     |

**Comparison:** otcfm DM (W2_scaled=1.90, r=0.737) is competitive with Klein TIGON (2.84, 0.874) — within ~1.5× on W2, slightly below on Pearson r. pdp DM (W2_scaled=11.74) is ~4–5× worse than Klein TIGON, indicating cord-blood is harder or the model needs tuning. Once all 24 cells are evaluated, a proper per-method Klein vs cord-blood multiplier can be computed.

---

## 7. Next Steps (Priority Order)

1. Fix `03_evaluate.py` argparser for otcfm and sf2m (add `--obsm_key`, `--n_dims`, `--tp_col`); re-trigger eval for 10 pending cells.
2. Resolve CUDA CUBLAS error for prescient eval (s7, s1234 DM; all PCA) — likely GPU contention; retry on fresh node.
3. Await job 29570505 (pdp PCA retrain) and run eval on completed checkpoints.
4. Await jobs 29570506-9 (deepruot + scDiffEq) and run eval pipeline on outputs.
5. Investigate pdp DM near-zero Pearson r — check if clone-proportion file, celltype_col, or k parameter needs adjustment for cord-blood dataset.
