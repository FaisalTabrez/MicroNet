#!/usr/bin/env Rscript
# run_spieceasi.R — Sparse inverse covariance network inference
#
# SPIEC-EASI (Sparse and Compositionally Robust Inference of Microbial Ecological Networks)
# Reference: Kurtz et al. (2015), PLoS Computational Biology
#
# Usage: Rscript run_spieceasi.R --input clr_matrix.tsv --outdir results/inference

suppressPackageStartupMessages({
  library(SpiecEasi)
  library(igraph)
  library(argparse)
  library(readr)
  library(dplyr)
})

# ── Argument parsing ─────────────────────────────────────────────────────────
parser <- ArgumentParser(description = "Run SPIEC-EASI network inference")
parser$add_argument("--input",  required = TRUE, help = "CLR abundance matrix (samples × taxa TSV)")
parser$add_argument("--outdir", default = "../../results/inference")
parser$add_argument("--method", default = "mb",
                    choices = c("mb", "glasso"),
                    help = "mb = Meinshausen-Bühlmann, glasso = graphical LASSO")
parser$add_argument("--rep-num", type = "integer", default = 50,
                    help = "StARS subsampling replicates (higher = more stable)")
parser$add_argument("--nlambda", type = "integer", default = 20,
                    help = "Number of regularization parameters")
args <- parser$parse_args()

dir.create(args$outdir, recursive = TRUE, showWarnings = FALSE)

# ── Load data ─────────────────────────────────────────────────────────────────
cat("Loading abundance matrix:", args$input, "\n")
mat <- read_tsv(args$input, show_col_types = FALSE)
sample_ids <- mat[[1]]
mat <- mat[, -1]                          # Remove sample ID column
otu_matrix <- as.matrix(mat)             # samples × taxa

cat(sprintf("Matrix: %d samples × %d taxa\n", nrow(otu_matrix), ncol(otu_matrix)))

# ── SPIEC-EASI ────────────────────────────────────────────────────────────────
cat(sprintf("Running SPIEC-EASI [method=%s, rep.num=%d] ...\n",
            args$method, args$rep_num))

# Note: SPIEC-EASI expects a count-like matrix (OTU counts), but can handle CLR
# by using the data.type="clr" option introduced in v1.1
se_result <- spiec.easi(
  otu_matrix,
  method           = args$method,
  lambda.min.ratio = 1e-2,
  nlambda          = args$nlambda,
  # FIX C4: the matrix is already CLR-transformed by clr_normalize.py.
  # Without data.type="clr", SPIEC-EASI applies its own CLR internally,
  # resulting in a double transformation that destroys the correction.
  data.type        = "clr",
  pulsar.params    = list(
    rep.num  = args$rep_num,
    ncores   = parallel::detectCores() - 1,
    thresh   = 0.05          # StARS stability threshold
  )
)

cat("Optimal lambda index:", se_result$select$opt.index, "\n")
cat("Lambda selected:", se_result$lambda[se_result$select$opt.index], "\n")

# ── Extract network ───────────────────────────────────────────────────────────
# Get weighted adjacency matrix (partial correlations)
adj_weight <- symBeta(getOptBeta(se_result), mode = "ave")
adj_binary <- getRefit(se_result)         # Binary adjacency (1/0)

taxa_names <- colnames(mat)
colnames(adj_weight) <- rownames(adj_weight) <- taxa_names
colnames(adj_binary) <- rownames(adj_binary) <- taxa_names

# ── Build igraph object ───────────────────────────────────────────────────────
# FIX C3: graph_from_adjacency_matrix(adj_weight) creates edges for every
# non-zero entry in the full dense matrix, including floating-point noise from
# the LASSO optimisation. Apply the binary SPIEC-EASI selection mask first so
# only statistically selected edges carry weight; everything else becomes 0.
adj_masked <- adj_weight * (adj_binary != 0)   # Zero out non-selected entries
g <- graph_from_adjacency_matrix(
  adj_masked,
  mode     = "undirected",
  weighted = TRUE,
  diag     = FALSE
)
V(g)$name <- taxa_names

cat(sprintf("Network: %d nodes, %d edges\n", vcount(g), ecount(g)))
cat(sprintf("Density: %.4f\n", graph.density(g)))

# ── Network statistics ─────────────────────────────────────────────────────────
stats <- data.frame(
  metric = c("n_nodes", "n_edges", "density", "avg_clustering",
             "mean_degree", "n_components", "largest_component_size"),
  value  = c(
    vcount(g), ecount(g), graph.density(g),
    mean(transitivity(g, type = "local"), na.rm = TRUE),
    mean(degree(g)),
    components(g)$no,
    max(components(g)$csize)
  )
)
print(stats)
write_tsv(stats, file.path(args$outdir, "spieceasi_network_stats.tsv"))

# ── Save outputs ──────────────────────────────────────────────────────────────
# Weighted adjacency matrix — FIX M1: save adj_masked, not adj_weight.
# adj_weight is the full dense partial-correlation matrix from the LASSO;
# it contains near-zero noise entries for every pair of taxa.
# adj_masked zeroes out all edges not selected by the StARS criterion, so the
# TSV matches exactly what the igraph object g contains.
# topology.py, train_vgae.py, and dashboard.py all read this TSV — they must
# see the same edge set as the R pipeline's own graph, not a denser one.
write_tsv(
  as.data.frame(as.matrix(adj_masked)) |> tibble::rownames_to_column("taxon"),
  file.path(args$outdir, "spieceasi_adj_weighted.tsv")
)

# Edge list with weights and signs
edges <- as_data_frame(g, what = "edges")
edges$sign <- ifelse(edges$weight > 0, "positive", "negative")
edges$abs_weight <- abs(edges$weight)
edges <- edges[order(-edges$abs_weight), ]
write_tsv(edges, file.path(args$outdir, "spieceasi_edgelist.tsv"))

# igraph object (for downstream use in R)
saveRDS(g, file.path(args$outdir, "spieceasi_graph.rds"))

cat("\nOutputs saved to:", args$outdir, "\n")

# ── SparCC comparison (optional baseline) ────────────────────────────────────
# FIX M5: SparCC applies its own internal log-ratio transformation and therefore
# requires RAW COUNT data as input. The otu_matrix here is CLR-transformed —
# feeding it to SparCC would produce a double transformation (same class of bug
# as C4 for SPIEC-EASI). Skip SparCC and direct the user to the raw counts.
cat("\nNOTE: SparCC comparison skipped — otu_matrix is CLR-transformed.\n")
cat("SparCC requires raw counts. Re-run with the pre-CLR count matrix to enable this comparison.\n")
