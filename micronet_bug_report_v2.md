# MicroNet — Second-Pass Bug & Continuity Audit

> Post-fix review of all 9 source files. Focuses on residual bugs, issues introduced by the fixes, cross-module continuity problems, and edge cases.

---

## Summary

| Severity | Count | Description |
|----------|-------|-------------|
| 🔴 Critical | 3 | Will crash or produce silently wrong results at runtime |
| 🟠 Major | 6 | Incorrect behavior that degrades scientific validity or breaks continuity |
| 🟡 Minor | 6 | Fragility, edge cases, or inconsistencies |

---

## 🔴 Critical Bugs

### C1. gLV Pseudo-Steady-State: LASSO with All-Zero Target Learns Only `r`, Never `A`
**File:** [glv_inference.py:272-274](file:///c:/Volume%20D/MicroNet/pipeline/02_inference/glv_inference.py#L268-L274)

**Bug:** The C2 fix set `dX_all = np.zeros_like(X_all)` for cross-sectional data. This is conceptually correct (steady-state: dx/dt = 0), but the downstream regression solves `(1/x_i) * dx_i/dt = r_i + A_i @ X`. With `dX_dt = 0` everywhere, `percap_growth` is all zeros, so the LASSO regression target `y` is a zero vector.

LASSO with an all-zero target and `fit_intercept=True` will learn `r_i ≈ 0` and `A_i ≈ 0` — the entire interaction matrix becomes zeros regardless of the actual data. **No interactions will ever be detected in cross-sectional mode.**

```python
# In fit_glv_lasso, line 84:
percap_growth = dX_dt / X_safe   # 0 / X_safe = 0 for every entry
# ...
model.fit(X, y)  # y is all zeros → coefs are all zeros → A = 0
```

**Fix:** For the pseudo-steady-state formulation, the regression should be restructured. At steady state: `0 = x_i * (r_i + Σ_j A_ij * x_j)`, so `r_i = -Σ_j A_ij * x_j` per sample. The proper regression is:

```python
# For cross-sectional pseudo-steady-state:
# Don't divide by x_i. Instead, regress x_i against all other x_j
# to capture A_ij relationships directly.
# The model becomes: for each sample s, x_i^s ≈ f(x_1^s, ..., x_N^s)
# Use negative self-abundance as the target:
for i in range(N):
    X_others = np.delete(X_all, i, axis=1)  # All taxa except i
    y = -X_all[:, i]  # Negative self: at SS, r_i = -sum A_ij*x_j
    model = LassoCV(...)
    model.fit(X_others, y)
    # Map coefficients back into the full A matrix
    A[i, :i] = model.coef_[:i]
    A[i, i+1:] = model.coef_[i:]
    r[i] = model.intercept_
```

Or better, use dedicated pseudo-steady-state methods from the literature (e.g., the LIMITS algorithm from Fisher & Mehta 2014).

> [!CAUTION]
> In the current code, running MicroNet on any cross-sectional dataset (no `time_point` column) will produce an all-zeros interaction matrix. Every downstream component (interaction classification, stability analysis, GNN edge labels, dashboard) will show zero or empty results.

---

### C2. VGAE Training Uses `train_data.train_pos_edge_index` — Attribute Doesn't Exist After `RandomLinkSplit`
**File:** [train_vgae.py:202](file:///c:/Volume%20D/MicroNet/pipeline/03_gnn/train_vgae.py#L197-L217)

**Bug:** The C5 fix correctly replaced `train_test_split_edges` with `RandomLinkSplit`, but the `train_epoch` function still references `data.train_pos_edge_index` (line 202) — an attribute created by the old API. `RandomLinkSplit` produces Data objects where the training edges are simply in `data.edge_index` and supervised edges are in `data.edge_label_index` / `data.edge_label`.

```python
def train_epoch(model, head, data, optimizer, device, beta=1.0):
    ...
    edge_index = data.train_pos_edge_index.to(device)  # ← AttributeError!
```

This will crash with `AttributeError: 'Data' object has no attribute 'train_pos_edge_index'` on the very first training step.

**Fix:**
```python
def train_epoch(model, head, data, optimizer, device, beta=1.0):
    model.train()
    optimizer.zero_grad()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)  # RandomLinkSplit stores training edges here

    z = model.encode(x, edge_index)
    loss = model.elbo(z, edge_index, beta=beta)
    # ... rest unchanged
```

---

### C3. SPIEC-EASI: `graph_from_adjacency_matrix` Includes Near-Zero Edges
**File:** [run_spieceasi.R:79-84](file:///c:/Volume%20D/MicroNet/pipeline/02_inference/run_spieceasi.R#L79-L84)

**Bug:** The M3 fix replaced `adj2igraph(adj_binary, ...)` with `graph_from_adjacency_matrix(adj_weight, ...)`. The problem is that `adj_weight` is the full dense weighted matrix from `symBeta(getOptBeta(...))`, which contains many near-zero entries (numerical noise from the optimization) that are NOT real edges. The old code used `adj_binary` (the thresholded refit) to decide which edges exist. The new code creates edges for **every non-zero entry** in the weighted matrix, including floating-point noise — potentially thousands of spurious edges.

The edge list saved on line 117 and all downstream analyses now include these ghost edges.

**Fix:** Zero out weights for edges not in the binary adjacency before building the graph:
```r
# Apply binary mask: keep only edges that SPIEC-EASI selected
adj_masked <- adj_weight * adj_binary   # Zero out non-selected edges
g <- graph_from_adjacency_matrix(
  adj_masked,
  mode     = "undirected",
  weighted = TRUE,
  diag     = FALSE
)
V(g)$name <- taxa_names
```

---

## 🟠 Major Bugs

### M1. VGAE `evaluate()` Function: Attribute Mismatch with `RandomLinkSplit` Output
**File:** [train_vgae.py:220-240](file:///c:/Volume%20D/MicroNet/pipeline/03_gnn/train_vgae.py#L220-L240)

**Bug:** The evaluate function was updated to use `val_data.edge_label_index` and `val_data.edge_label` (correct for `RandomLinkSplit`), but the encoding step uses `val_data.edge_index`:

```python
z = model.encode(x, edge_index)  # line 225 — val_data.edge_index
```

With `RandomLinkSplit`, `val_data.edge_index` contains the *training* message-passing edges (so encoding can access the graph structure during validation). This is actually correct behavior for message-passing — the encoder should use training edges, not validation edges. **This is fine.**

However, lines 229-230 split `edge_label_index` by `edge_label`:
```python
pos_edge = val_data.edge_label_index[:, val_data.edge_label == 1].to(device)
neg_edge = val_data.edge_label_index[:, val_data.edge_label == 0].to(device)
```

`RandomLinkSplit` with default `neg_sampling_ratio=1.0` adds negative samples, so `edge_label` contains both 1s and 0s. But if `add_negative_train_samples=False` was set (line 272), negative samples are **only omitted from the training split**, not from val/test. So this should work for val_data. **The logic is fine for validation, but the following issue exists:**

The evaluate function signature changed from `evaluate(model, data, device)` to `evaluate(model, val_data, device)`, which is correct. ✅ No bug here on closer inspection.

**However**, there IS a related bug: after the final model is loaded (line 337), the code encodes using `train_data.edge_index`, which is correct. But `data.taxa_names` is used on line 342 — `data` is the original pre-split Data object. If `RandomLinkSplit` modifies the node set (it doesn't for link prediction), this could break. In practice this is safe, but fragile.

**Downgrading to informational.** No action needed.

---

### M1 (revised). `config.yaml` Paths Are Relative — But Relative to What?
**File:** [config.yaml:4-5](file:///c:/Volume%20D/MicroNet/pipeline/01_profiling/config.yaml#L4-L5)

**Bug:** The config specifies:
```yaml
data_dir: "../../data"
results_dir: "../../results"
```

The comment says "Paths are relative to the project root", but Snakemake resolves relative paths **from the Snakefile's directory** (`pipeline/01_profiling/`), not from where you invoke it. The README's quick-start says:
```bash
snakemake -s pipeline/01_profiling/Snakefile --cores 8
```
When invoked from the project root, `-s pipeline/01_profiling/Snakefile` sets the working directory to `pipeline/01_profiling/`, so `../../data` resolves to `data/` at the project root — **this is correct**.

**But**: if anyone invokes Snakemake from `pipeline/01_profiling/` directly (e.g., `cd pipeline/01_profiling && snakemake`), the paths are still relative to the Snakefile location — still correct. **However**, the `glob.glob` on line 13 of the Snakefile:
```python
SAMPLES = [Path(f).stem.replace("_1", "")
           for f in glob.glob(str(DATA_DIR / "qc" / "*_1.fastq.gz"))]
```
uses `DATA_DIR = Path(config["data_dir"])`, which is `../../data`. In Snakemake, `Path` works relative to the CWD, not the Snakefile. If invoked from the project root via `-s`, the CWD is the project root, so `../../data` resolves to **two levels above the project root**, which is wrong.

**Fix:** Use Snakemake's `workflow.basedir` or absolute paths:
```python
# At top of Snakefile, after configfile:
SNAKEFILE_DIR = Path(workflow.basedir)
DATA_DIR      = (SNAKEFILE_DIR / Path(config["data_dir"])).resolve()
RESULTS_DIR   = (SNAKEFILE_DIR / Path(config["results_dir"])).resolve()
```

---

### M2. VGAE: Missing `--abundance` and `--adjacency` in README Quick Start
**File:** [README.md:70](file:///c:/Volume%20D/MicroNet/README.md#L69-L70)

**Bug:** The README's VGAE command is:
```bash
python pipeline/03_gnn/train_vgae.py --epochs 200
```

But `--abundance` and `--adjacency` are **required** arguments in `train_vgae.py` (lines 247-248). Running this command will crash with `argparse` error.

**Fix:**
```bash
python pipeline/03_gnn/train_vgae.py \
    --abundance results/profiling/clr_abundance_matrix.tsv \
    --adjacency results/inference/spieceasi_adj_weighted.tsv \
    --interactions results/inference/classified_interactions.tsv \
    --epochs 200
```

---

### M3. Topology → Dashboard Continuity: `pagerank` Column Missing from Dashboard Display
**File:** [dashboard.py:280](file:///c:/Volume%20D/MicroNet/pipeline/05_viz/dashboard.py#L279-L282) ↔ [topology.py:114](file:///c:/Volume%20D/MicroNet/pipeline/04_analysis/topology.py#L114)

**Bug:** The M5 fix replaced `authority_score` with `pagerank` in `topology.py`'s centrality computation. But the dashboard's Keystone Taxa tab still displays:
```python
cent_df.head(30)[["degree", "betweenness", "eigenvector",
                   "closeness", "keystone_score"]].round(4)
```

This doesn't include `pagerank` or `hub_score` — two of the seven metrics that feed into `keystone_score`. Users can't see which centrality components are driving the ranking. More importantly, the `topology.py` print statement on line 326 still references `eigenvector` which exists, so it's fine — but `pagerank` is invisible in the dashboard.

**Fix:** Add `pagerank` to the displayed columns:
```python
cent_df.head(30)[["degree", "betweenness", "eigenvector",
                   "closeness", "pagerank", "keystone_score"]].round(4)
```

---

### M4. Robustness Simulation: `lcc_fraction` Uses Original Node Count as Denominator
**File:** [topology.py:235](file:///c:/Volume%20D/MicroNet/pipeline/04_analysis/topology.py#L229-L238)

**Bug:**
```python
"lcc_fraction": lcc_size / len(nodes),
```

`len(nodes)` is the **original** node count, not the current graph size. After removing 50% of nodes, an LCC of 50% of the *remaining* nodes would report as 25% — which makes it look like the community is more fragile than it actually is. Ecologically, robustness is typically reported as LCC / original_N, so this is defensible. But the dashboard label says "Fraction of community remaining" which is ambiguous.

**Not a bug per se**, but the fraction at step 0 (before any removal) will already be < 1.0 if the graph has disconnected components. This could confuse users.

**Fix (clarity):** Rename the dashboard label or add a note:
```python
labels={"lcc_fraction": "LCC / original network size"},
```

---

### M5. SparCC Runs on CLR Data — SparCC Expects Raw Counts
**File:** [run_spieceasi.R:126](file:///c:/Volume%20D/MicroNet/pipeline/02_inference/run_spieceasi.R#L124-L127)

**Bug:** SparCC (`sparcc()` in SpiecEasi) is designed for **count data** (or at most relative abundances). The input `otu_matrix` is now confirmed to be CLR-transformed data (per the C4 fix adding `data.type = "clr"` for SPIEC-EASI). SparCC applies its own log-ratio transformation internally, so feeding it CLR data means **double log-ratio transformation** — the exact same class of bug that C4 fixed for SPIEC-EASI, but left unfixed for SparCC.

```r
sparcc_result <- sparcc(otu_matrix)  # ← CLR data, but SparCC expects counts
```

The SparCC results (`sparcc_correlations.tsv`) will contain invalid correlation values, and the edge comparison on line 146 is meaningless.

**Fix:** Either:
1. Load raw count data separately for SparCC, or
2. Remove the SparCC comparison (it's labeled "optional baseline"), or
3. Add a clear comment/warning and skip SparCC when input is CLR:

```r
cat("\nWARNING: Skipping SparCC comparison — input is CLR-transformed.\n")
cat("SparCC requires raw counts. To run SparCC, pass the pre-CLR count matrix.\n")
```

---

### M6. gLV Heatmap: `taxa[:30]` Truncation Causes Dimension Mismatch with A
**File:** [glv_inference.py:308](file:///c:/Volume%20D/MicroNet/pipeline/02_inference/glv_inference.py#L307-L309)

**Bug:**
```python
plot_interaction_matrix(A, taxa[:30] if len(taxa) > 30 else taxa,
                        output_path=...)
```

`A` is the full `(N, N)` matrix but `taxa[:30]` is only 30 labels. `sns.heatmap` will plot the full matrix with only 30 tick labels, causing a visual misalignment — labels won't correspond to the correct rows/columns. If N=200, the first label corresponds to row 0 but visually appears to be row ~3.

**Fix:** Slice the matrix too:
```python
n_show = min(30, len(taxa))
plot_interaction_matrix(A[:n_show, :n_show], taxa[:n_show],
                        output_path=str(outdir / "interaction_matrix_heatmap.png"))
```

---

## 🟡 Minor Bugs

### m1. Snakefile: `zcat` Not Available on Windows
**File:** [Snakefile:68](file:///c:/Volume%20D/MicroNet/pipeline/01_profiling/Snakefile#L68)

**Bug:** The m2 fix moved the temp file from `/tmp` to a Snakemake `temp()` output (good), but the `zcat` command is still used:
```bash
zcat {input.r1} {input.r2} > {output.cat_reads}
```

`zcat` is a Unix utility. On Windows (the user's OS), this will fail. The conda bioconda tools run through WSL or conda-provided binaries, but `zcat` itself may not be in the conda environment's PATH.

**Fix:** Use `gzip -dc` which is more portable, or use Python:
```bash
gzip -dc {input.r1} {input.r2} > {output.cat_reads}
```

---

### m2. Dashboard: `show_gnn` Checkbox Has No Effect
**File:** [dashboard.py:229](file:///c:/Volume%20D/MicroNet/pipeline/05_viz/dashboard.py#L229)

**Bug:** The sidebar has a `show_gnn` checkbox:
```python
show_gnn = st.sidebar.checkbox("Include GNN-predicted edges", value=False)
```

But `show_gnn` is never passed to `build_nx_graph()` or used anywhere. The GNN-predicted edges are never included regardless of the checkbox state. The `build_nx_graph()` function doesn't even accept a `gnn_probs` parameter.

**Fix:** Wire the checkbox to the graph builder:
```python
def build_nx_graph(data: dict, min_weight: float = 0.0,
                   include_gnn: bool = False, gnn_threshold: float = 0.7) -> nx.Graph:
    # ... existing adjacency code ...
    if include_gnn and "gnn_probs" in data:
        prob_df = data["gnn_probs"]
        for t1 in adj.index:
            for t2 in adj.columns:
                if t1 >= t2:
                    continue
                p = prob_df.loc[t1, t2] if (t1 in prob_df.index and t2 in prob_df.columns) else 0
                if p >= gnn_threshold and not G.has_edge(t1, t2):
                    G.add_edge(t1, t2, weight=p, source="gnn",
                               interaction_type="Predicted", sign="unknown",
                               guild_i=-1, guild_j=-1, is_inter_guild=False)
    return G

# In main():
G = build_nx_graph(data, min_weight=min_weight, include_gnn=show_gnn)
```

---

### m3. Topology: `is_small_world` Can Be `None` → `print()` Formats Badly
**File:** [topology.py:351](file:///c:/Volume%20D/MicroNet/pipeline/04_analysis/topology.py#L351)

**Bug:**
```python
print(f"Small-world sigma: {sw['sigma']:.3f} → {'YES' if sw['is_small_world'] else 'NO'}")
```

If `sigma` is `nan`, the `:3f` format prints `nan` correctly. But `is_small_world` is `None` (not `False`) when sigma is nan (line 276). `None` is falsy, so it prints "NO" — which is misleading. The result is "inconclusive", not "no".

**Fix:**
```python
sw_label = "YES" if sw["is_small_world"] is True else ("NO" if sw["is_small_world"] is False else "INCONCLUSIVE")
print(f"Small-world sigma: {sw['sigma']:.3f} → {sw_label}")
```

---

### m4. CLR Transform: Unused `gmean` Import
**File:** [clr_normalize.py:14](file:///c:/Volume%20D/MicroNet/pipeline/01_profiling/clr_normalize.py#L14)

```python
from scipy.stats import gmean  # Never used
```

The CLR transform correctly computes `log(x) - mean(log(x))` (which equals `log(x / geometric_mean(x))`), but the explicit `gmean` import is dead code.

**Fix:** Remove the import:
```python
# from scipy.stats import gmean  # Not needed — CLR uses log-space arithmetic
```

---

### m5. gLV `classify_interactions`: Interaction with One Side Below Threshold Falls Through
**File:** [glv_inference.py:148-154](file:///c:/Volume%20D/MicroNet/pipeline/02_inference/glv_inference.py#L148-L154)

**Bug:** If `a_ij = 0.005` and `a_ji = 0.5` (threshold = 0.01), then:
- `s_ij = "0"` (below threshold)  
- `s_ji = "+"` (above threshold)

The overall signature `"0/+"` maps to `"Commensalism"`. The pair passes the gate on line 149 because `abs(a_ji) >= threshold`. This is correct.

**But**: if `a_ij = 0.005` and `a_ji = -0.005`, both are below threshold. The pair is **skipped** (line 150). This is also correct.

**Edge case**: if `a_ij = 0.02` and `a_ji = 0.02`, signature is `"+/+"` → Mutualism. But with a strength of 0.02, this is likely noise. There's no **minimum strength filter** — very weak interactions that barely exceed the threshold are classified with the same confidence as strong ones.

**Fix (enhancement):** Add a minimum strength filter:
```python
if abs(a_ij) < threshold and abs(a_ji) < threshold:
    continue
strength = (abs(a_ij) + abs(a_ji)) / 2
if strength < threshold:  # Additional: skip if average is below threshold
    continue
```

---

### m6. Environment: `torch-scatter` and `torch-sparse` Version Compatibility
**File:** [environment.yml:50-51](file:///c:/Volume%20D/MicroNet/pipeline/00_setup/environment.yml#L50-L51)

**Bug:** `torch-scatter==2.1.2` and `torch-sparse==0.6.18` must be compiled against the exact same PyTorch and CUDA versions. Installing via pip without specifying the `--find-links` URL for the correct CUDA/CPU build will either:
1. Fail to compile (requires a C++ toolchain), or
2. Install a CPU-only build even if CUDA is available

**Fix:** Add the PyG wheel index:
```yaml
- pip:
    - torch==2.1.0
    - --find-links https://data.pyg.org/whl/torch-2.1.0+cu121.html
    - torch-geometric==2.4.0
    - torch-scatter==2.1.2
    - torch-sparse==0.6.18
```

Or use the unified PyG install:
```yaml
- pip:
    - torch==2.1.0
    - torch-geometric==2.4.0
    - pyg-lib  # Handles scatter/sparse automatically
```

---

## Cross-Module Continuity Issues

### Continuity Matrix

| Upstream Output | Consumer | Column/Key Expected | Status |
|---|---|---|---|
| `clr_normalize.py` → `clr_abundance_matrix.tsv` | `run_spieceasi.R` | First column = sample ID (unnamed) | ⚠️ R reads with `mat[[1]]`; CLR writes `index_col=0` name as `sample_id`. If `read_tsv` treats `sample_id` as a data column, `mat[,-1]` drops it correctly. **OK but fragile.** |
| `clr_normalize.py` → `clr_abundance_matrix.tsv` | `glv_inference.py` | `pd.read_csv(index_col=0)` | ✅ Correct |
| `run_spieceasi.R` → `spieceasi_adj_weighted.tsv` | `train_vgae.py` | `adj_df.reindex(index=taxa, columns=taxa)` | ⚠️ `taxa` comes from CLR matrix column names; adj uses `taxa_names` from R. If taxon name formatting differs (e.g., underscores vs spaces), `reindex` fills with NaN → empty graph. |
| `run_spieceasi.R` → `spieceasi_adj_weighted.tsv` | `topology.py` | `adj_df.index` = taxon names | ✅ Correct |
| `glv_inference.py` → `classified_interactions.tsv` | `train_vgae.py` | `taxon_i_name`, `taxon_j_name`, `interaction_type` | ✅ Correct |
| `glv_inference.py` → `classified_interactions.tsv` | `topology.py` | Same columns | ✅ Correct |
| `topology.py` → `centrality_metrics.tsv` | `dashboard.py` | Expects `keystone_score`, `betweenness`, `eigenvector`, `closeness` columns | ⚠️ Dashboard doesn't display `pagerank` (see M3) |
| `topology.py` → `topology_summary.tsv` | `dashboard.py` | Expects keys: `sigma`, `C_real`, `C_random`, `r_squared`, `power_law_exponent`, `L_real` | ⚠️ `is_small_world` is `None` → saved as empty string → `float()` will crash on "None" string. See below. |
| `train_vgae.py` → `predicted_edge_probabilities.tsv` | `topology.py` | `prob_df.loc[t1, t2]` | ⚠️ Taxon name alignment (same risk as above) |

### Critical Continuity Issue: `is_small_world` = `None` Crashes Dashboard

`topology.py` line 276:
```python
"is_small_world": sigma > 1 if not np.isnan(sigma) else None,
```

`pd.Series(topo_summary).to_csv()` serializes `None` as an empty string. The dashboard doesn't try to read `is_small_world`, so it won't crash there. But `sigma` itself can be `nan`, and `pd.Series.to_csv()` serializes `nan` as an empty string too. On the dashboard side:
```python
float(topo.loc['sigma', 'value'])  # float("") → ValueError
```

**Fix:** Handle NaN gracefully in the dashboard:
```python
def safe_float(topo, key, default=0.0):
    try:
        return float(topo.loc[key, 'value'])
    except (ValueError, KeyError):
        return default

st.metric("Small-world σ", f"{safe_float(topo, 'sigma'):.3f}")
```

---

## Priority Fix Order

1. **C2** — `train_pos_edge_index` AttributeError crashes VGAE training immediately
2. **C1** — All-zero gLV on cross-sectional data produces empty interaction matrix
3. **C3** — Spurious edges from unmasked weighted adjacency inflate network size
4. **M5** — SparCC on CLR data is double-transformed (same class as original C4)
5. **M6** — Heatmap label/matrix dimension mismatch
6. **M1** — Snakemake path resolution from different working directories
7. **M2** — README missing required VGAE arguments
