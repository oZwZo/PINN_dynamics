# Method-Agnostic Trajectory Inference Evaluation

This document specifies the evaluation protocol implemented in
`scripts/fate_eval_pipeline.py` and consumed by all
`scripts/<method>/03_evaluate.py` runners (PRESCIENT, TIGON,
pseudodynamics+, TrajectoryNet, OT-CFM, SF2M, MIOFlow). The protocol
is method-agnostic: each method only needs to provide a
`simulation_func` that maps `(start_cells, model, n_sims, device)` to
endpoint coordinates in the same embedding space used at training.

---

## 1. Inputs

| Input | Type | Description |
|---|---|---|
| `adata` | `AnnData` | Full dataset with `obs[tp_col]`, `obs[well_col]`, `obs[celltype_col]`, `obs.clones`, and `obsm[obsm_key]` |
| `F_obs` | `DataFrame` (cells × cell_types) | Clone-traced ground-truth fate proportions, indexed by cell barcode |
| `clone_proportions` | `DataFrame` | Index = clone IDs used for W2 evaluation (test-set clones) |
| `model` | method-specific | Trained trajectory model |
| `simulation_func` | callable | `(start_cells, model, n_sims, device) -> ndarray` returning `(n_cells, n_sims, n_dims)` (stochastic) or `(n_cells, n_dims)` (deterministic) |
| `scaler` | `{mean, std}` or `None` | Source is **method-dependent** and must reproduce the *exact* inverse used at training. See §2.4. |

The split is fixed by `well_col`: `Well == 2` is **test**, the rest is
**train**. Three timepoints `tp_first < tp_mid < tp_last` are inferred
from `sorted(adata.obs[tp_col].unique())`.

---

## 2. Metrics, data sources, and embedding space

All metrics operate in `adata.obsm[obsm_key][:, :n_dims]`. When a model
is trained on standardized embeddings, the *same* standardized space is
used for fate prediction (KNN was fit there); only W2 is additionally
reported in raw space via `inverse_standardize`.

### 2.1 Fate accuracy and Pearson r

| Aspect | Source |
|---|---|
| **Start cells** | `adata` cells at `tp_first` whose barcode is in `F_obs.index` (any `Well`) |
| **Start time** | `tp_first` (earliest timepoint, e.g. day 2 in Klein) |
| **End time** | `tp_last` (terminal timepoint, e.g. day 6) |
| **KNN reference** | Training cells (`Well != 2`); features = `obsm[obsm_key][:, :n_dims]`, labels = `obs[celltype_col]` |
| **Space** | Embedding space used at training (`obsm_key`, e.g. `X_pca` or `DM_EigenVectors`) |

**Procedure**

1. Replicate each start cell `n_sims` times and forward-simulate to
   `tp_last` via `simulation_func`. Stochastic models produce
   `n_sims` independent endpoints per start cell; deterministic models
   produce a single endpoint (and `n_sims` is ignored).
2. Apply the fitted `KNeighborsClassifier(k)` to every endpoint to
   assign a cell-type label.
3. Aggregate labels per start cell into a fate distribution
   `F_hat ∈ R^{n_cells × n_cell_types}` (rows sum to 1, or all-zero).
4. Restrict `F_obs` to the same cells/columns →
   `F_obs_aligned`.
5. **Accuracy** = fraction of cells where
   `argmax(F_hat row) == argmax(F_obs row)`. Cells with an all-zero
   `F_hat` row are labelled `"Undifferentiated"`.
6. **Pearson r** = correlation between column-mean vectors of
   `F_obs_aligned` and `F_hat` (population-level distribution
   similarity).

### 2.2 Population Wasserstein-2

| Aspect | Source |
|---|---|
| **Start cells** | Test cells (`Well == 2`) at `tp_mid`, restricted to clones in `clone_proportions.index` |
| **Target cells** | Test cells at `tp_last`, same clone restriction |
| **Start time** | `tp_mid` |
| **End time** | `tp_last` |
| **Space** | Same `obsm_key[:, :n_dims]` slice; both standardized and raw if `scaler` provided |

**Procedure**

1. Forward-simulate source → endpoints; flatten any `n_sims` axis so
   endpoints are `(n_cells * n_sims, n_dims)`.
2. `w2_scaled = compute_w2(endpoints, target)` (entropic OT,
   `torchcfm.wasserstein`, `reg=0.05`).
3. If `scaler` is provided,
   `w2_raw = compute_w2(inverse_standardize(endpoints), inverse_standardize(target))`;
   else `w2_raw = w2_scaled`.

> **`w2_raw` is the canonical metric for cross-method comparison.**
> Methods trained in raw PC space (e.g. PRESCIENT) report identical
> `w2_scaled` and `w2_raw`; methods trained on standardized embeddings
> (e.g. SF2M, OT-CFM, pseudodynamics+) must inverse-standardize before
> the values are comparable.

### 2.4 Choosing the scaler

The scaler must reproduce the *exact* inverse of whatever standardization
was applied at training time. **There is no one-size-fits-all rule**:

| Training input | Correct scaler source |
|---|---|
| Raw obsm (e.g. PRESCIENT reads `X_pca`, pdp+ reads `DM_EigenVectors`) | `scaler = None`. `w2_raw == w2_scaled`, both already in raw PC space. |
| Raw obsm + on-the-fly standardize with cohort-specific stats (sf2m, otcfm: train-only mean/std over `X_pca`) | Load the `scaler.npz` saved at training time. **Do not** recompute via `compute_scaler_from_adata` — full-population stats differ from train-only stats and are not a valid inverse. |
| Pre-standardized obsm whose unscaled counterpart exists in adata and was generated as full-population standardize (e.g. pdp+ with `cellstate_key='X_pca_scaled'` derived from `X_pca` over all cells) | `compute_scaler_from_adata(adata, cellstate_key, n_dims)`. |

If none of these apply (pre-scaled obsm with unknown stats), there is
no reliable inverse and `w2_raw` should be reported as `None`.

### 2.3 Per-clone Wasserstein-2

| Aspect | Source |
|---|---|
| **Start cells** | For each shared clone: test cells at `tp_mid` in that clone |
| **Target cells** | For each shared clone: test cells at `tp_last` in that clone |
| **Iteration** | `clones in (src_clone_ids ∩ target_clone_ids)` |

For each clone, repeat the population-W2 procedure on its sub-population
and append `{clone, w2_scaled, w2_raw, clone_size_src, clone_size_tgt}`.

---

## 3. Output format

Written to `output_dir` (default = `model_dir`):

| File | Schema |
|---|---|
| `eval_combined.csv` | one row: `sim_mode, accuracy, pearson_r, w2_scaled, w2_raw, n_start_cells, n_sims_fate, k` |
| `F_hat_{mode}.csv` | `(n_cells × n_cell_types)`, index = cell barcode, columns = cell types, values = predicted proportion |
| `w2_per_clone_{mode}.csv` | columns: `clone, w2_scaled, w2_raw, clone_size_src, clone_size_tgt` |

`{mode}` identifies the simulation regime (e.g. `sde`, `ode`, `sb`) so
multiple regimes for the same model can coexist.

---

## 4. Adding a new method

1. Provide a closure
   `sim_fn(start_cells, model, n_sims, device) -> ndarray` (use the
   `make_*_sim_fn` factories in `fate_eval_pipeline.py` as templates).
2. Build a `03_evaluate.py` runner that:
   - loads `adata`, `F_obs`, `clone_proportions`, the trained `model`,
     and (if applicable) a `scaler` via `compute_scaler_from_adata`;
   - assembles `start_cells_fate`, `start_cell_ids`, `x_ref/y_ref`,
     `src_cells_w2/tgt_cells_w2`, `src_clones_w2/tgt_clones_w2` exactly
     as described in §2;
   - calls
     `run_fate_evaluation`, `compute_w2_population`, and
     `compute_w2_per_clone` from `fate_eval_pipeline`;
   - writes the three CSVs in §3.

No metric code is duplicated across methods — only data assembly and
the `simulation_func` differ.
