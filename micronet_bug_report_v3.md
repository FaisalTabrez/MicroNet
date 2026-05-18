# MicroNet — Third-Pass Audit (v3)

> Final review of all 10 source files after 39 cumulative fixes from v1 and v2. Focused on regression from fixes, remaining edge cases, and end-to-end data flow integrity.

---

## Overall Assessment

The codebase is in **good shape**. All critical and major bugs from v1 and v2 are correctly resolved. The remaining issues are edge cases, a consistency gap in the SPIEC-EASI output, and hardening improvements. No crash-level bugs remain on the primary iHMP workflow path.

| Severity | Count | Description |
|----------|-------|-------------|
| 🟠 Moderate | 1 | Silent data inconsistency between saved file and live graph |
| 🟡 Low | 3 | Edge-case crashes or inefficiencies |
| 🔵 Informational | 4 | Hardening, style, or documentation nits |

---

## 🟠 Moderate Issues

### M1. SPIEC-EASI: Saves **Unmasked** `adj_weight` but Graph Uses **Masked**
**File:** [run_spieceasi.R:80 vs 110](file:///c:/Volume%20D/MicroNet/pipeline/02_inference/run_spieceasi.R#L80-L112)

**Bug:** The C3 fix correctly masks the adjacency before building the igraph:
```r
adj_masked <- adj_weight * (adj_binary != 0)   # Line 80 — correct
g <- graph_from_adjacency_matrix(adj_masked, ...)
```

But the **saved TSV** on line 110 still writes the **unmasked** `adj_weight`:
```r
write_tsv(
  as.data.frame(as.matrix(adj_weight)) |> ...    # ← UNMASKED
  file.path(args$outdir, "spieceasi_adj_weighted.tsv")
)
```

This means:
- The igraph `g` (used for edge list, stats, RDS) has only selected edges ✅
- The TSV `spieceasi_adj_weighted.tsv` has the **full dense** matrix with noise entries ❌
- `topology.py` and `train_vgae.py` and `dashboard.py` all read the TSV → they will see ghost edges that the R script's own graph doesn't contain

This is the same class of inconsistency that C3 was meant to fix, but the fix only applied to the igraph construction and missed the TSV export.

**Fix:** Save `adj_masked` instead of `adj_weight`:
```r
write_tsv(
  as.data.frame(as.matrix(adj_masked)) |> tibble::rownames_to_column("taxon"),
  file.path(args$outdir, "spieceasi_adj_weighted.tsv")
)
```

> [!IMPORTANT]
> This is the highest-priority item. Every Python module downstream reads this TSV. Until it's fixed, `topology.py`, `train_vgae.py`, and `dashboard.py` all operate on a denser graph than the R pipeline intended, potentially inflating edge counts and diluting keystone scores.

---

## 🟡 Low Issues

### L1. gLV Time-Series: Subjects with ≤1 Timepoint Cause Empty Arrays
**File:** [glv_inference.py:309-316](file:///c:/Volume%20D/MicroNet/pipeline/02_inference/glv_inference.py#L309-L317)

**Bug:** When grouping by `subject_id`, if a subject has only one timepoint, `compute_derivatives()` returns arrays of shape `(0, N)`:
```python
dX, X_mid = compute_derivatives(X, times)  # X is (1, N) → dX is (0, N)
all_X.append(X_mid)   # (0, N) — empty
all_dX.append(dX)     # (0, N) — empty
```

These empty arrays silently pass through `np.vstack()`, adding zero rows. This doesn't crash but wastes iteration and logs misleading sample counts. If **all** subjects have only one timepoint, `X_all` is `(0, N)` → the LASSO fit crashes.

**Fix:**
```python
for subj, grp in meta.groupby("subject_id"):
    grp = grp.sort_values("time_point")
    if len(grp) < 2:
        print(f"  Skipping subject {subj}: only {len(grp)} timepoint(s)")
        continue
    X = clr.loc[grp.index].values
    times = grp["time_point"].values.astype(float)
    dX, X_mid = compute_derivatives(X, times)
    all_X.append(X_mid)
    all_dX.append(dX)

if not all_X:
    print("ERROR: No subjects with ≥2 timepoints found. Cannot fit temporal gLV.")
    print("Falling back to pseudo-steady-state.")
    A, r = fit_glv_pss(clr.values, method=args.method)
    dX_all = None  # Skip the normal branch
else:
    X_all = np.vstack(all_X)
    dX_all = np.vstack(all_dX)
```

---

### L2. Snakefile: `conda:` Directive Points to Non-Existent `../../envs/micronet.yml`
**File:** [Snakefile:36, 54, 69, 104](file:///c:/Volume%20D/MicroNet/pipeline/01_profiling/Snakefile#L36)

**Bug:** Every rule references:
```python
conda: "../../envs/micronet.yml"
```

But the actual environment file is at `pipeline/00_setup/environment.yml`. The `envs/` directory doesn't exist. When running with `--use-conda`, Snakemake will fail to find this file and crash.

**Fix:** Point to the actual file:
```python
conda: "../00_setup/environment.yml"
```

Or create the `envs/micronet.yml` symlink/copy for the expected path.

---

### L3. `test_scale_free`: Crashes on Graphs Where All Nodes Have the Same Degree
**File:** [topology.py:280-302](file:///c:/Volume%20D/MicroNet/pipeline/04_analysis/topology.py#L280-L302)

**Bug:** If every node has the same degree (e.g., a regular graph or ring), `Counter(degrees)` has one key, so `k_vals` has length 1. `stats.linregress` requires at least 2 points and will raise a `ValueError`.

**Fix:**
```python
if len(k_vals) < 2:
    return {
        "power_law_exponent": float("nan"),
        "r_squared": 0.0,
        "p_value": 1.0,
        "is_scale_free": False,
    }
```

---

## 🔵 Informational / Hardening

### I1. `fit_glv_pss`: Diagonal Self-Regulation is Heuristic
**File:** [glv_inference.py:169](file:///c:/Volume%20D/MicroNet/pipeline/02_inference/glv_inference.py#L169)

```python
A[i, i] = -np.abs(coefs).mean()
```

This sets the diagonal (self-regulation / carrying capacity) to the negative mean of off-diagonal coefficients. This is a reasonable heuristic but not derived from the LIMITS algorithm. The original LIMITS paper (Fisher & Mehta 2014) doesn't estimate diagonals this way. This won't crash but should be noted in the docstring as a heuristic, not a literature-derived formula.

**Suggestion:** Add a comment or make it configurable:
```python
# Heuristic: self-regulation proportional to average interaction strength.
# The LIMITS algorithm does not prescribe a specific diagonal estimator.
A[i, i] = -np.abs(coefs).mean() if len(coefs) > 0 else -1.0
```

---

### I2. VGAE Loss Plot X-Axis Label is Wrong After M7 Fix
**File:** [train_vgae.py:326](file:///c:/Volume%20D/MicroNet/pipeline/03_gnn/train_vgae.py#L326)

```python
ax1.set_title("ELBO Loss"); ax1.set_xlabel("Epoch (×10)")
```

After the M7 fix, `history["loss"]` now has one entry per epoch (not per 10 epochs). The x-axis label "Epoch (×10)" is misleading — each point is now a single epoch.

**Fix:** `ax1.set_xlabel("Epoch")`

---

### I3. Topology: `compute_centralities` Returns Partial Data for Non-LCC Nodes
**File:** [topology.py:86-130](file:///c:/Volume%20D/MicroNet/pipeline/04_analysis/topology.py#L86-L130)

`betweenness` and `closeness` are computed on the LCC only, but `degree`, `eigenvector`, `hub_score`, `pagerank`, and `strength` are computed on the full graph. The `.fillna(0)` on line 120 fills in zeros for non-LCC nodes' betweenness/closeness, which is semantically correct (nodes outside the LCC have zero betweenness from the LCC's perspective). However, this means those nodes' keystone scores are deflated by two zero metrics, which could mislead users into thinking isolated taxa have very low ecological importance.

**Suggestion:** Add a note to the output or the dashboard indicating that betweenness/closeness are NaN for disconnected taxa.

---

### I4. Dashboard: Interaction Type Pie Shows Counts from **All** gLV Interactions, Not Just Displayed Edges
**File:** [dashboard.py:280-286](file:///c:/Volume%20D/MicroNet/pipeline/05_viz/dashboard.py#L279-L286)

```python
itype_counts = data["interactions"]["interaction_type"].value_counts()
```

This counts interaction types from the **full** classified interactions file, regardless of the `min_weight` filter or `max_nodes` limit applied to the visible graph. The pie chart may show interaction types that have been filtered out of the network view, creating a visual disconnect.

**Suggestion:** Compute pie counts from the visible graph edges instead:
```python
if G.number_of_edges() > 0:
    visible_types = [d["interaction_type"] for _, _, d in G.edges(data=True)]
    itype_counts = pd.Series(visible_types).value_counts()
```

---

## Cross-Module Data Flow: Final Verification

| Step | Producer | Consumer | Column/Key Match | Status |
|------|----------|----------|-------------------|--------|
| 1 | `clr_normalize.py` → `clr_abundance_matrix.tsv` | `run_spieceasi.R` | `mat[[1]]` = sample_id col | ✅ |
| 2 | `clr_normalize.py` → `clr_abundance_matrix.tsv` | `glv_inference.py` | `pd.read_csv(index_col=0)` | ✅ |
| 3 | `clr_normalize.py` → taxon names use spaces (`str.replace("_", " ")`) | `run_spieceasi.R` → taxon names = `colnames(mat)` | Both read the same TSV → identical | ✅ |
| 4 | `run_spieceasi.R` → `spieceasi_adj_weighted.tsv` | `train_vgae.py` → `adj_df.reindex(taxa)` | Taxa from CLR columns = adj row/col names | ✅ (both from same TSV) |
| 5 | `run_spieceasi.R` → `spieceasi_adj_weighted.tsv` | `topology.py` → `adj_df.index` | Same TSV | ✅ |
| 6 | `run_spieceasi.R` → `spieceasi_adj_weighted.tsv` | `dashboard.py` → `data["adj"]` | Same TSV | ⚠️ **See M1** — unmasked |
| 7 | `glv_inference.py` → `classified_interactions.tsv` | `train_vgae.py` → `taxon_i_name`, `taxon_j_name` | Uses CLR column names (spaces) | ✅ |
| 8 | `glv_inference.py` → `classified_interactions.tsv` | `topology.py` → same columns | ✅ |
| 9 | `topology.py` → `centrality_metrics.tsv` | `dashboard.py` → expects `pagerank` | ✅ (added in M3 v2 fix) |
| 10 | `topology.py` → `topology_summary.tsv` | `dashboard.py` → `safe_float()` | ✅ (NaN-safe) |
| 11 | `train_vgae.py` → `predicted_edge_probabilities.tsv` | `topology.py` → `prob_df.loc[t1, t2]` | Uses CLR column names | ✅ |
| 12 | `train_vgae.py` → `predicted_edge_probabilities.tsv` | `dashboard.py` → `build_nx_graph(include_gnn=True)` | ✅ (wired in m2 v2 fix) |
| 13 | `glv_inference.py` → `dX_all = None` sentinel | `fit_glv_pss()` | Dispatches correctly | ✅ |
| 14 | `config.yaml` → `data_dir`, `results_dir` | Snakefile → `SNAKEFILE_DIR / ...` | Resolves via `workflow.basedir` | ✅ |

---

## Priority Fix Order

1. **M1** — Save masked adjacency to TSV (highest impact — affects three downstream consumers)
2. **L2** — Fix conda env path in Snakefile (blocks `--use-conda` execution)
3. **L1** — Skip single-timepoint subjects (prevents crash on sparse longitudinal data)
4. **L3** — Guard `linregress` for uniform-degree graphs
5. **I2** — Fix loss plot x-axis label (cosmetic but misleading)
