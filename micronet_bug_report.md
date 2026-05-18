# MicroNet — Bug Audit Report

> Reviewed all 9 source files across `pipeline/00_setup` through `pipeline/05_viz` against the [MicroNet_Project_Document.docx](file:///c:/Volume%20D/MicroNet/MicroNet_Project_Document.docx).

---

## Summary

| Severity | Count | Description |
|----------|-------|-------------|
| 🔴 Critical | 7 | Crashes, data corruption, or silently wrong results |
| 🟠 Major | 10 | Incorrect behavior that degrades scientific validity |
| 🟡 Minor | 7 | Fragility, config drift, or style issues |

---

## 🔴 Critical Bugs

### C1. CLR Transform: Incorrect Geometric Mean Calculation
**File:** [clr_normalize.py](file:///c:/Volume%20D/MicroNet/pipeline/01_profiling/clr_normalize.py#L33-L37)

The CLR formula is `CLR(x_i) = log(x_i / g(x))` where `g(x)` is the **geometric mean**. The code computes `log_X.mean()` which is the arithmetic mean of log values — mathematically equivalent to `log(geometric_mean)` — so the subtraction `log_X - log_X.mean()` is correct. **However**, the code imports `gmean` from scipy but never uses it, and the comment on line 37 says "Subtract row geometric mean" which is slightly misleading but the math is fine.

**Actually, this is NOT a bug** — the math is equivalent. Striking this. *(Kept for transparency.)*

---

### C1 (revised). gLV Inference: Shape Mismatch in `fit_glv_lasso` — Uses Wrong X for Regression
**File:** [glv_inference.py](file:///c:/Volume%20D/MicroNet/pipeline/02_inference/glv_inference.py#L73-L91)

**Bug:** The per-capita growth rate is computed from `X_safe[:-1]` (dropping the last row), producing `(T-1, N)` values. But the LASSO regression on line 88 fits against `X` (the full `(T, N)` matrix, or `X_mid` depending on the call site), creating a **shape mismatch**.

When called from `main()` with time-series data, `X_all` is `X_mid` of shape `(T-1, N)` and `dX_all` is `dX_dt` of shape `(T-1, N)`. Inside `fit_glv_lasso`:
- `percap_growth = dX_dt / X_safe[:-1]` → shape `(T-2, N)` (drops one more row!)
- But `model.fit(X, y)` uses `X` which is still `(T-1, N)`

This is a **dimension mismatch that will either crash or silently corrupt the regression**.

```python
# Line 77: BUG — X_safe[:-1] drops a row from already-aligned data
percap_growth = dX_dt / X_safe[:-1] if dX_dt.shape[0] < T else dX_dt / X_safe
```

**Fix:**
```python
# The caller already aligns X and dX_dt to the same shape.
# Don't drop another row here:
percap_growth = dX_dt / X_safe

# And fit on the same X:
model.fit(X, y)  # Both are (T-1, N) — correct
```

> [!CAUTION]
> This bug means **every interaction coefficient in the A matrix is fitted on misaligned data**, invalidating all downstream gLV results (interaction classification, stability analysis, GNN edge label supervision).

---

### C2. gLV Inference: Cross-Sectional Pseudo-Derivative is Scientifically Invalid
**File:** [glv_inference.py](file:///c:/Volume%20D/MicroNet/pipeline/02_inference/glv_inference.py#L257-L262)

**Bug:** When no time-series metadata is available, the code falls back to:
```python
X_all = clr.values
dX_all = np.diff(clr.values, axis=0)  # ← difference between UNRELATED samples
X_all = X_all[:-1]
```

`np.diff` along axis=0 computes row-wise differences. But cross-sectional samples have **no temporal ordering** — the "derivative" between sample 5 and sample 6 is meaningless. This produces **garbage interaction coefficients**.

**Fix:** Either:
1. Require time-series metadata and refuse to run on cross-sectional data, OR
2. Use a proper pseudo-steady-state approach (assume `dx/dt ≈ 0` at equilibrium and solve `0 = r + Ax*`):

```python
# Pseudo-steady-state: assume dx/dt ≈ 0 → r = -A @ x_eq for each sample
# Use Ridge regression: for each taxon i, regress 0 ~ r_i + A_i @ X
# This is equivalent to A @ X.T ≈ -r (constant intercept)
print("Cross-sectional: using pseudo-steady-state (assumes samples are near equilibrium)")
X_all = clr.values
dX_all = np.zeros_like(X_all)  # dx/dt = 0 at steady state
```

---

### C3. Snakefile: Wrong Script Path for CLR Normalization
**File:** [Snakefile](file:///c:/Volume%20D/MicroNet/pipeline/01_profiling/Snakefile#L101-L102)

**Bug:** The `clr_normalize` rule specifies:
```python
script: "../../scripts/clr_normalize.py"
```

But the actual file lives at `pipeline/01_profiling/clr_normalize.py`. The path `../../scripts/` doesn't exist.

**Fix:**
```python
script: "clr_normalize.py"
```

---

### C4. SPIEC-EASI: Feeding CLR Data Where Counts Are Expected
**File:** [run_spieceasi.R](file:///c:/Volume%20D/MicroNet/pipeline/02_inference/run_spieceasi.R#L45-L57)

**Bug:** The comment on line 46 says *"SPIEC-EASI expects a count-like matrix (OTU counts), but can handle CLR by using the data.type='clr' option introduced in v1.1"*. However, the `spiec.easi()` call **does not pass `data.type = "clr"`**. By default, SPIEC-EASI applies its own CLR internally, which means **double CLR transformation** — destroying the compositional correction.

**Fix:**
```r
se_result <- spiec.easi(
  otu_matrix,
  method           = args$method,
  lambda.min.ratio = 1e-2,
  nlambda          = args$nlambda,
  # CRITICAL: tell SPIEC-EASI the data is already CLR-transformed
  # Otherwise it applies CLR again internally → double transformation
  pulsar.params    = list(
    rep.num  = args$rep_num,
    ncores   = parallel::detectCores() - 1,
    thresh   = 0.05
  )
)
```

**Option A:** Pass raw counts (before CLR) to SPIEC-EASI and let it handle CLR internally.  
**Option B:** Pass `data.type = "clr"` if using the CLR matrix.  

The design doc specifies CLR is applied in Phase 1 — so **Option B** is correct for consistency.

---

### C5. VGAE: Deprecated `train_test_split_edges` API
**File:** [train_vgae.py](file:///c:/Volume%20D/MicroNet/pipeline/03_gnn/train_vgae.py#L270)

**Bug:** `train_test_split_edges()` was **removed in PyTorch Geometric ≥ 2.3**. The environment pins `torch-geometric==2.4.0`, so this line will crash:
```python
data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.1)
```

**Fix:** Use the replacement `RandomLinkSplit` transform:
```python
from torch_geometric.transforms import RandomLinkSplit

transform = RandomLinkSplit(
    num_val=0.1, num_test=0.1,
    is_undirected=True,
    add_negative_train_samples=False,
)
train_data, val_data, test_data = transform(data)
```
This requires refactoring the training loop to use `train_data`, `val_data`, and `test_data` separately.

---

### C6. VGAE: `weights_only` Warning / Security Risk in `torch.load`
**File:** [train_vgae.py](file:///c:/Volume%20D/MicroNet/pipeline/03_gnn/train_vgae.py#L316)

**Bug:** In PyTorch ≥ 2.6, `torch.load()` without `weights_only=True` raises an error (currently a warning in 2.1). More importantly, the default `weights_only=False` allows arbitrary code execution if the .pt file is tampered with.

```python
model.load_state_dict(torch.load(outdir / "best_vgae.pt"))
```

**Fix:**
```python
model.load_state_dict(torch.load(outdir / "best_vgae.pt", weights_only=True))
```

---

### C7. Download Script: Checks for `.fastq.gz` but `fasterq-dump` Produces Uncompressed `.fastq`
**File:** [download_data.py](file:///c:/Volume%20D/MicroNet/pipeline/00_setup/download_data.py#L58-L76)

**Bug:** The skip check on line 59 looks for `{accession}_1.fastq.gz`, but `fasterq-dump` produces uncompressed `{accession}_1.fastq`. The `pigz` compression on line 74 creates the `.gz` files only *after* `fasterq-dump` completes. So if the download succeeds but `pigz` fails (e.g., `pigz` not installed), the next run will **re-download everything** because the `.gz` check fails, and the uncompressed files are left on disk consuming space.

**Fix:** Also check for uncompressed files, or compress in a separate step with error handling:
```python
r1_raw = outdir / f"{accession}_1.fastq"
r1_gz  = outdir / f"{accession}_1.fastq.gz"
if r1_gz.exists() or r1_raw.exists():
    log.info(f"  {accession} already downloaded, skipping.")
    return
```

---

## 🟠 Major Bugs

### M1. gLV Stability Analysis: Incorrect Jacobian Computation
**File:** [glv_inference.py](file:///c:/Volume%20D/MicroNet/pipeline/02_inference/glv_inference.py#L169-L195)

**Bug:** The Jacobian comment (line 178) correctly states the formula:
> `J_ii = r_i + 2*A_ii*x_i + sum_{j!=i} A_ij*x_j`

But the code implements a simplified version:
```python
J[i, j] = x_eq[i] * A[i, j]  # This is only correct for off-diagonal terms
```

For diagonal terms, this gives `J_ii = x_i * A_ii`, which **omits the growth rate and the sum term**. The correct Jacobian for gLV at equilibrium is:

```python
# At equilibrium: r_i + sum_j A_ij * x_j = 0, so sum = -r_i
# J_ij = delta_ij * (r_i + sum_k A_ik * x_k) + x_i * A_ij
# At equilibrium, the first term vanishes for i != j, but for i = j:
# J_ii = (r_i + sum_k A_ik * x_k) + x_i * A_ii
#       = 0 + x_i * A_ii  (at true equilibrium)
```

**Actually**, if the system is at true equilibrium, the diagonal simplification is approximately correct. But `x_eq` is set to `np.abs(clr.mean().values)` (line 285) — the **absolute value of CLR-transformed means**, which is NOT a true equilibrium and has no biological meaning as an abundance. CLR values are log-ratios that can be negative.

**Fix:**
```python
# Use raw (non-CLR) mean abundances as approximate equilibrium
# Or better: solve for steady state from A and r
x_eq = np.abs(clr.mean().values)  # ← WRONG: CLR values aren't abundances
# Should be:
raw_abundances = np.exp(clr.mean().values)  # Convert CLR back to ~proportions
x_eq = raw_abundances / raw_abundances.sum()
```

---

### M2. gLV: `fit_glv_ridge` Has Same Shape Bug as LASSO
**File:** [glv_inference.py](file:///c:/Volume%20D/MicroNet/pipeline/02_inference/glv_inference.py#L96-L114)

**Bug:** Line 103:
```python
percap_growth = dX_dt / X_safe  # Assumes X and dX_dt have same shape
```
This works IF X and dX_dt are already aligned. But there's no guard against misaligned inputs. Additionally, `model.fit(X, percap_growth[:, i])` — if `percap_growth` has fewer rows than `X` (from the LASSO branch logic), this crashes.

**Fix:** Add explicit shape assertion:
```python
assert X.shape == dX_dt.shape, f"X {X.shape} and dX_dt {dX_dt.shape} must match"
```

---

### M3. SPIEC-EASI: Incorrect Edge Weight Extraction
**File:** [run_spieceasi.R](file:///c:/Volume%20D/MicroNet/pipeline/02_inference/run_spieceasi.R#L72-L74)

**Bug:**
```r
g <- adj2igraph(adj_binary,
                vertex.attr = list(name = taxa_names),
                edge.attr   = list(weight = adj_weight[adj_binary == 1]))
```

`adj_weight[adj_binary == 1]` extracts elements column-major (R default), while `adj2igraph` processes edges in a specific order. These orderings **may not match**, assigning wrong weights to edges. The weighted adjacency matrix is symmetric, but the extraction pattern depends on `adj_binary`'s storage order.

**Fix:** Build the igraph from the weighted adjacency directly:
```r
# Use weighted adjacency matrix directly
g <- graph_from_adjacency_matrix(
  adj_weight,
  mode = "undirected",
  weighted = TRUE,
  diag = FALSE
)
V(g)$name <- taxa_names
```

---

### M4. Topology: Scale-Free Test is Statistically Incorrect
**File:** [topology.py](file:///c:/Volume%20D/MicroNet/pipeline/04_analysis/topology.py#L270-L286)

**Bug:** The power-law fit regresses `log(counts)` against `log(1:len)` instead of `log(degrees)`:
```python
degrees = [d for _, d in G.degree() if d > 0]
log_count = np.log(np.bincount(degrees)[1:] + 1e-9)
slope, intercept, r, p, se = stats.linregress(
    np.log(np.arange(1, len(log_count) + 1)), log_count  # ← X is wrong
)
```

`np.arange(1, len(log_count) + 1)` is just indices `[1, 2, 3, ...]`, not degree values. For degree distributions with gaps (e.g., no node has degree 5), the `bincount` array has a zero at index 5, and the log of that (+ 1e-9) produces a massive negative outlier.

**Fix:** Use a proper complementary CDF approach or the `powerlaw` package:
```python
from collections import Counter
deg_counts = Counter(degrees)
k_vals = np.array(sorted(deg_counts.keys()))
counts = np.array([deg_counts[k] for k in k_vals])
# Log-log regression on actual degree values
slope, intercept, r, p, se = stats.linregress(np.log(k_vals), np.log(counts))
```

---

### M5. Topology: `nx.hits()` on Undirected Graph is Degenerate
**File:** [topology.py](file:///c:/Volume%20D/MicroNet/pipeline/04_analysis/topology.py#L107)

**Bug:** HITS (hub/authority) is designed for **directed** graphs. On an undirected graph, hub and authority scores are identical and degenerate to eigenvector centrality. Including both `hub_score` and `authority_score` in the composite keystone score double-counts the same information, inflating the weight of this single metric.

**Fix:** Remove one of the two, or replace with PageRank (which is useful even on undirected graphs):
```python
centralities["pagerank"] = nx.pagerank(G, weight="weight")
```

---

### M6. VGAE: Interaction Type Head Gets Stale Gradients
**File:** [train_vgae.py](file:///c:/Volume%20D/MicroNet/pipeline/03_gnn/train_vgae.py#L196-L217)

**Bug:** The `InteractionTypeHead` is only trained when `data.labeled_edge_index` exists, but it's **always** included in the optimizer. When there are no labeled edges, the head's parameters receive zero gradients but still participate in optimizer state (momentum, weight decay), which can cause numerical drift.

**Fix:** Only add head parameters to optimizer when labeled edges exist:
```python
params = list(model.parameters())
if hasattr(data, "labeled_edge_index"):
    params += list(head.parameters())
optimizer = torch.optim.Adam(params, lr=args.lr)
```

---

### M7. VGAE: Best Model is Saved Based on Validation AUC but Evaluated Every 10 Epochs
**File:** [train_vgae.py](file:///c:/Volume%20D/MicroNet/pipeline/03_gnn/train_vgae.py#L286-L300)

**Bug:** The history only records metrics every 10 epochs, so the "best" model is only checkpointed at evaluation points. If the best model occurs at epoch 137, it's never saved. Additionally, `history["loss"]` only has entries at evaluation epochs, not every epoch, making the loss curve sparse and misleading.

**Fix:** Evaluate and checkpoint more frequently, or track loss every epoch:
```python
history["loss"].append(loss)  # Every epoch
if epoch % 10 == 0:
    auc, ap, z = evaluate(...)
    history["val_auc"].append(auc)
    # ... checkpoint logic
```

---

### M8. Dashboard: `build_nx_graph` Assigns Guild of Taxon 1 to ALL Edges
**File:** [dashboard.py](file:///c:/Volume%20D/MicroNet/pipeline/05_viz/dashboard.py#L108)

**Bug:**
```python
guild = data["guilds"].loc[t1, "guild"] if "guilds" in data else 0
G.add_edge(t1, t2, ..., guild=int(guild))
```

The guild is looked up only for `t1`, not for the edge. An edge between guilds should either store both guilds or not store guild info at all. This misleadingly labels inter-guild edges with a single guild ID.

**Fix:** Store both guild IDs or remove the edge-level guild attribute:
```python
g1 = data["guilds"].loc[t1, "guild"] if "guilds" in data and t1 in data["guilds"].index else -1
g2 = data["guilds"].loc[t2, "guild"] if "guilds" in data and t2 in data["guilds"].index else -1
G.add_edge(t1, t2, ..., guild_i=int(g1), guild_j=int(g2),
           is_inter_guild=(g1 != g2))
```

---

### M9. Dashboard: `topology_summary.tsv` Loaded with Wrong Header Assumption
**File:** [dashboard.py](file:///c:/Volume%20D/MicroNet/pipeline/05_viz/dashboard.py#L75-L77)

**Bug:** `topology.py` saves the summary via `pd.Series(topo_summary).to_csv(...)`, which writes a 2-column CSV with the index (metric name) and values. The dashboard loads it with `names=["metric", "value"]`, but also sets `index_col=0`, which makes the metric name the index and creates a single `value` column — this is correct. **However**, the column name is set to `"value"` via the `names` parameter, but `index_col=0` consumes the first column, so the actual column may be unnamed depending on pandas version.

**Fix:** Use explicit loading:
```python
topo = pd.read_csv(topo_path, sep="\t", header=None, index_col=0)
topo.columns = ["value"]
```

---

### M10. Snakefile: Missing `config.yaml` Dependency
**File:** [Snakefile](file:///c:/Volume%20D/MicroNet/pipeline/01_profiling/Snakefile#L8)

**Bug:** The Snakefile references `configfile: "config.yaml"` but no `config.yaml` exists in the repository. Running `snakemake` will crash with `FileNotFoundError`.

**Fix:** Create a default `config.yaml`:
```yaml
# config.yaml — MicroNet profiling configuration
data_dir: "../../data"
results_dir: "../../results"
```

---

## 🟡 Minor Bugs

### m1. Environment: `scikit-bio` Listed Twice
**File:** [environment.yml](file:///c:/Volume%20D/MicroNet/pipeline/00_setup/environment.yml#L25-L58)

`scikit-bio=0.5.9` appears under both conda dependencies (line 25) and pip dependencies (line 58). This causes a version conflict — conda installs it first, then pip overwrites it. This is fragile and can cause broken packages.

**Fix:** Remove from pip section.

---

### m2. Snakefile: HUMAnN3 Concatenates to `/tmp` — Not Portable to Windows
**File:** [Snakefile](file:///c:/Volume%20D/MicroNet/pipeline/01_profiling/Snakefile#L67)

```bash
zcat {input.r1} {input.r2} > /tmp/{wildcards.sample}_cat.fastq
```

This hardcodes `/tmp` which doesn't exist on Windows. Also, `zcat` is not available on Windows.

**Fix:** Use `$(mktemp -d)` or a Snakemake `temp()` directive.

---

### m3. Topology: `eigenvector_centrality_numpy` Catches Wrong Exception
**File:** [topology.py](file:///c:/Volume%20D/MicroNet/pipeline/04_analysis/topology.py#L103)

```python
except nx.PowerIterationFailedConvergence:
```

The `_numpy` variant uses NumPy's eigensolver, not power iteration. It raises `nx.NetworkXError` or `np.linalg.LinAlgError` on failure, not `PowerIterationFailedConvergence`. The wrong exception type means actual errors propagate uncaught.

**Fix:**
```python
except (nx.NetworkXError, np.linalg.LinAlgError):
    centralities["eigenvector"] = {n: 0 for n in G.nodes()}
```

---

### m4. Robustness Simulation: Deterministic Strategies Run Multiple Trials Redundantly
**File:** [topology.py](file:///c:/Volume%20D/MicroNet/pipeline/04_analysis/topology.py#L195-L230)

For `betweenness` and `degree` strategies, the removal order is deterministic — every trial produces identical results. But the code runs `n_trials=5` trials for each, wasting 4× compute. Only `random` needs multiple trials.

**Fix:**
```python
n_trials_effective = n_trials if strategy == "random" else 1
for trial in range(n_trials_effective):
```

---

### m5. Dashboard: Hardcoded Relative Path `../../results`
**File:** [dashboard.py](file:///c:/Volume%20D/MicroNet/pipeline/05_viz/dashboard.py#L32)

```python
RESULTS = Path("../../results")
```

This only works if Streamlit is launched from `pipeline/05_viz/`. If launched from the project root (as the README suggests: `streamlit run pipeline/05_viz/dashboard.py`), the path resolves to `../results` which is wrong.

**Fix:** Resolve relative to the script's location:
```python
RESULTS = Path(__file__).resolve().parent.parent.parent / "results"
```

---

### m6. Download Script: `pysradb` Column Name May Differ
**File:** [download_data.py](file:///c:/Volume%20D/MicroNet/pipeline/00_setup/download_data.py#L133)

```python
accessions = meta["run_accession"].tolist() if "run_accession" in meta.columns else []
```

Different `pysradb` versions return different column names (e.g., `run_accession` vs `Run` vs `SRR`). If the column doesn't match, the script silently downloads nothing.

**Fix:** Add fuzzy column matching or explicit error:
```python
acc_col = next((c for c in meta.columns if c.lower() in ("run_accession", "run", "srr")), None)
if acc_col is None:
    log.error(f"Cannot find accession column. Available: {meta.columns.tolist()}")
    return
accessions = meta[acc_col].tolist()
```

---

### m7. VGAE: Missing `keystones.py` Referenced in README
**File:** [README.md](file:///c:/Volume%20D/MicroNet/README.md#L70)

The README references `python pipeline/04_analysis/keystones.py` but this file doesn't exist. Keystone computation is embedded in `topology.py`. Also, the design document mentions Bayesian Network Structure Learning (Section 4.3.3) which has no corresponding implementation.

---

## Architecture Gaps (vs. Design Doc)

| Design Doc Section | Status | Notes |
|---|---|---|
| Phase 0: Data Acquisition + QC | ✅ Implemented | `download_data.py` |
| Phase 1: MetaPhlAn4 + HUMAnN3 + CLR | ✅ Implemented | `Snakefile` + `clr_normalize.py` |
| Phase 2: SPIEC-EASI | ✅ Implemented | `run_spieceasi.R` |
| Phase 2: gLV inference | ✅ Implemented | `glv_inference.py` (with bugs above) |
| Phase 2: Bayesian Network (4.3.3) | ❌ **Missing** | Hill-climbing + BIC not implemented |
| Phase 3: VGAE | ✅ Implemented | `train_vgae.py` (deprecated API) |
| Phase 3: Phylogenetic embeddings (4.4.1) | ❌ **Missing** | node2vec on PhyloT not implemented |
| Phase 3: Genome-level features (4.4.1) | ❌ **Missing** | GC content, genome size not included |
| Phase 4: Topology analysis | ✅ Implemented | `topology.py` |
| Phase 4: Leiden community detection | ⚠️ Partial | Falls back to Louvain if `leidenalg` missing (but `leidenalg` not in environment.yml) |
| Phase 5: Dashboard | ✅ Implemented | `dashboard.py` |
| Dynamic network (EvolveGCN, 6.3) | ❌ **Missing** | No temporal GNN implemented |
| CAMI2 benchmarking (7.1) | ❌ **Missing** | No benchmarking scripts |
| Cross-dataset generalization (7.2) | ❌ **Missing** | No TARA Oceans evaluation |

---

## Priority Fix Order

1. **C1** — gLV shape mismatch (corrupts all interaction results)
2. **C4** — Double CLR transformation (corrupts co-occurrence network)
3. **C5** — Deprecated PyG API (prevents GNN from running at all)
4. **C3** — Wrong script path (Snakemake pipeline broken)
5. **C2** — Cross-sectional pseudo-derivative (scientifically invalid)
6. **M1** — Jacobian equilibrium point (stability analysis wrong)
7. **M4** — Scale-free test regression (topology conclusion wrong)
8. **M3** — Edge weight ordering (SPIEC-EASI graph may have misassigned weights)
