# MicroNet: Microbial Ecological Network Reconstruction

> Inferring cooperation, competition, and symbiosis from metagenomics data
> using co-occurrence analysis, Bayesian ecological inference, and Graph Neural Networks.

## Project Structure

```
micronet/
├── pipeline/
│   ├── 00_setup/           # Environment setup & dependency checks
│   ├── 01_profiling/       # Taxonomic & functional profiling
│   ├── 02_inference/       # Co-occurrence + Bayesian interaction inference
│   ├── 03_gnn/             # Graph Neural Network training & link prediction
│   ├── 04_analysis/        # Network topology, keystone taxa, guild detection
│   └── 05_viz/             # Interactive visualization dashboard
├── data/                   # Downloaded datasets
├── results/                # Output networks, models, figures
└── docs/                   # Documentation
```

## Recommended Datasets

### Primary: Human Microbiome Project Phase 2 (iHMP)
- **URL**: https://hmpdacc.org/ihmp/
- **SRA BioProject**: PRJNA398089
- **What**: Longitudinal multi-omic data from IBD, T2D, and pregnancy cohorts
- **Size**: ~2,500 metagenome samples
- **Why**: Temporal data enables dynamic network reconstruction with gLV

### Secondary: Earth Microbiome Project (EMP)
- **URL**: https://earthmicrobiome.org/
- **SRA BioProject**: PRJEB13870
- **What**: 16S amplicon data from 96 distinct biomes worldwide
- **Size**: ~27,751 samples
- **Why**: Cross-environment comparison of network topology

### Benchmarking: CAMI2 Synthetic Communities
- **URL**: https://data.cami-challenge.org/
- **What**: Simulated metagenomes with known ground-truth community composition
- **Why**: Validate your interaction inference against known interactions

### Supplementary: TARA Oceans
- **URL**: https://www.ebi.ac.uk/metagenomics/studies/MGYS00002008
- **What**: Marine surface water metagenomes, global scale
- **Why**: Environmental context for cross-biome network comparison

## Quick Start

```bash
# 1. Set up conda environment
conda env create -f pipeline/00_setup/environment.yml
conda activate micronet

# 2. Download data
python pipeline/00_setup/download_data.py --dataset hmp --n-samples 100

# 3. Run profiling
snakemake -s pipeline/01_profiling/Snakefile --cores 8

# 4. Infer interactions
python pipeline/02_inference/glv_inference.py \
    --abundance results/profiling/clr_abundance_matrix.tsv \
    --metadata data/hmp/metadata.tsv

Rscript pipeline/02_inference/run_spieceasi.R \
    --input results/profiling/clr_abundance_matrix.tsv

# 5. Train GNN
# FIX M2: --abundance and --adjacency are required arguments; omitting them
# causes an immediate argparse error. All key paths are shown explicitly.
python pipeline/03_gnn/train_vgae.py \
    --abundance results/profiling/clr_abundance_matrix.tsv \
    --adjacency results/inference/spieceasi_adj_weighted.tsv \
    --interactions results/inference/classified_interactions.tsv \
    --epochs 200

# 6. Analyze network (keystone taxa are computed inside topology.py)
python pipeline/04_analysis/topology.py \
    --adjacency results/inference/spieceasi_adj_weighted.tsv \
    --interactions results/inference/classified_interactions.tsv

# 7. Launch dashboard
streamlit run pipeline/05_viz/dashboard.py
```
