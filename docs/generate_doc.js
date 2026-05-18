const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, LevelFormat, TableOfContents,
} = require("docx");
const fs = require("fs");

// ── Color palette ─────────────────────────────────────────────────────────
const TEAL   = "1A7A6E";
const DARK   = "1C2B2D";
const ACCENT = "2ECC71";
const LIGHT  = "EAF5F3";
const MID    = "D0EDE9";
const GRAY   = "F5F5F5";

// ── Border helpers ─────────────────────────────────────────────────────────
const border = (color = "CCCCCC") => ({ style: BorderStyle.SINGLE, size: 1, color });
const borders = (c = "CCCCCC") => ({ top: border(c), bottom: border(c), left: border(c), right: border(c) });

// ── Paragraph helpers ──────────────────────────────────────────────────────
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, font: "Arial", size: 32, bold: true, color: TEAL })],
    spacing: { before: 360, after: 160 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: TEAL, space: 6 } },
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, font: "Arial", size: 26, bold: true, color: DARK })],
    spacing: { before: 280, after: 120 },
  });
}

function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    children: [new TextRun({ text, font: "Arial", size: 22, bold: true, color: TEAL })],
    spacing: { before: 200, after: 80 },
  });
}

function body(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, font: "Arial", size: 22, color: DARK, ...opts })],
    spacing: { after: 120, line: 280 },
  });
}

function bullet(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "bullets", level },
    children: [new TextRun({ text, font: "Arial", size: 22, color: DARK })],
    spacing: { after: 80 },
  });
}

function callout(text, bgColor = LIGHT) {
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [9360],
    rows: [
      new TableRow({
        children: [
          new TableCell({
            borders: {
              top:    { style: BorderStyle.SINGLE, size: 12, color: TEAL },
              bottom: { style: BorderStyle.SINGLE, size: 1,  color: MID  },
              left:   { style: BorderStyle.SINGLE, size: 12, color: TEAL },
              right:  { style: BorderStyle.SINGLE, size: 1,  color: MID  },
            },
            shading: { fill: bgColor, type: ShadingType.CLEAR },
            margins: { top: 120, bottom: 120, left: 200, right: 120 },
            children: [new Paragraph({
              children: [new TextRun({ text, font: "Arial", size: 20, color: DARK, italics: true })],
              spacing: { after: 0 },
            })],
          }),
        ],
      }),
    ],
    margins: { top: 160, bottom: 160 },
  });
}

function spacer() {
  return new Paragraph({ children: [new TextRun("")], spacing: { after: 80 } });
}

// ── Table helpers ──────────────────────────────────────────────────────────
function headerCell(text, width, bg = TEAL) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    borders: borders(TEAL),
    shading: { fill: bg, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({
      children: [new TextRun({ text, font: "Arial", size: 19, bold: true, color: "FFFFFF" })],
      alignment: AlignmentType.CENTER,
      spacing: { after: 0 },
    })],
  });
}

function dataCell(text, width, shade = false) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    borders: borders("DDDDDD"),
    shading: { fill: shade ? LIGHT : "FFFFFF", type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({
      children: [new TextRun({ text, font: "Arial", size: 19, color: DARK })],
      spacing: { after: 0 },
    })],
  });
}

// ── Title page ─────────────────────────────────────────────────────────────
function titleSection() {
  return [
    new Paragraph({ children: [new TextRun("")], spacing: { after: 1440 } }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "MicroNet", font: "Arial", size: 72, bold: true, color: TEAL })],
      spacing: { after: 120 },
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({
        text: "Microbial Ecological Network Reconstruction",
        font: "Arial", size: 36, bold: false, color: DARK,
      })],
      spacing: { after: 240 },
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      border: { top: { style: BorderStyle.SINGLE, size: 4, color: ACCENT }, bottom: { style: BorderStyle.SINGLE, size: 4, color: ACCENT } },
      children: [new TextRun({
        text: "Inferring Cooperation, Competition, and Symbiosis from Metagenomics Data",
        font: "Arial", size: 22, italics: true, color: "555555",
      })],
      spacing: { before: 160, after: 160 },
    }),
    new Paragraph({ children: [new TextRun("")], spacing: { after: 720 } }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Project Design Document  |  Version 1.0", font: "Arial", size: 20, color: "888888" })],
      spacing: { after: 80 },
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "May 2026", font: "Arial", size: 20, color: "888888" })],
      spacing: { after: 2880 },
    }),
    new Paragraph({ children: [new TextRun({ break: 1 })] }),
  ];
}

// ── Document body ──────────────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          { level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
          { level: 1, format: LevelFormat.BULLET, text: "\u25E6", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
        ],
      },
      {
        reference: "numbers",
        levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
        ],
      },
    ],
  },
  styles: {
    default: {
      document: { run: { font: "Arial", size: 22 } },
    },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: TEAL },
        paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: DARK },
        paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: "Arial", color: TEAL },
        paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 2 } },
    ],
  },
  sections: [
    // ── Title Page ─────────────────────────────────────────────────────────
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        },
      },
      children: titleSection(),
    },
    // ── Main Document ──────────────────────────────────────────────────────
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1260, bottom: 1440, left: 1260 },
        },
      },
      headers: {
        default: new Header({
          children: [
            new Paragraph({
              children: [
                new TextRun({ text: "MicroNet — Microbial Ecological Network Reconstruction", font: "Arial", size: 16, color: "888888" }),
              ],
              border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: TEAL } },
              spacing: { after: 0 },
            }),
          ],
        }),
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              alignment: AlignmentType.RIGHT,
              border: { top: { style: BorderStyle.SINGLE, size: 4, color: TEAL } },
              children: [
                new TextRun({ text: "Page ", font: "Arial", size: 16, color: "888888" }),
                new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 16, color: "888888" }),
                new TextRun({ text: " of ", font: "Arial", size: 16, color: "888888" }),
                new TextRun({ children: [PageNumber.TOTAL_PAGES], font: "Arial", size: 16, color: "888888" }),
              ],
              spacing: { before: 0 },
            }),
          ],
        }),
      },
      children: [

        // ── Table of Contents ──────────────────────────────────────────────
        h1("Table of Contents"),
        new TableOfContents("Table of Contents", {
          hyperlink: true,
          headingStyleRange: "1-3",
          stylesWithLevels: [
            { styleName: "Heading 1", level: 1 },
            { styleName: "Heading 2", level: 2 },
            { styleName: "Heading 3", level: 3 },
          ],
        }),
        spacer(),

        // ─── 1. Executive Summary ──────────────────────────────────────────
        h1("1. Executive Summary"),
        body("MicroNet is a computational metagenomics project that reconstructs ecological interaction networks from microbial community sequencing data. By combining three complementary inference strategies — sparse co-occurrence analysis, generalized Lotka-Volterra (gLV) dynamical modeling, and Graph Neural Networks (GNNs) — MicroNet produces a signed, weighted ecological graph encoding cooperation, competition, and symbiosis between microbial taxa."),
        spacer(),
        body("The project addresses a fundamental challenge in microbial ecology: observational data (who is present, how abundant) does not directly reveal interactions (who helps or harms whom). MicroNet bridges this gap by treating interaction inference as a machine learning problem over graph-structured biological data."),
        spacer(),
        callout("Core question: In a microbial community, which organisms help each other grow, which compete for resources, and which exploit others — and how does this interaction web change across environments, disease states, or time?"),
        spacer(),

        // ─── 2. Scientific Background ──────────────────────────────────────
        h1("2. Scientific Background"),

        h2("2.1 Why Microbial Ecological Networks?"),
        body("Microbial communities drive virtually every biogeochemical process on Earth — from nitrogen fixation in soil to vitamin synthesis in the human gut. Despite their importance, the ecological rules governing these communities remain poorly understood. Network-theoretic approaches offer a powerful framework: by representing taxa as nodes and interactions as edges, we can apply decades of ecological and graph theory to understand community structure, stability, and function."),
        spacer(),
        body("Microbial ecological networks have direct applications in:"),
        bullet("Clinical medicine — identifying keystone gut taxa whose disruption drives IBD, obesity, or antibiotic-associated diarrhea"),
        bullet("Agriculture — engineering soil microbiome networks to improve crop yields"),
        bullet("Biotechnology — designing stable, productive synthetic communities for fermentation"),
        bullet("Environmental monitoring — tracking ecosystem health via network topology"),
        spacer(),

        h2("2.2 The Compositionality Problem"),
        body("Metagenomics data is compositional: relative abundances sum to one, so an increase in one taxon mathematically decreases others — even with no biological interaction. Naively computing Pearson or Spearman correlations on relative abundances produces spurious negative correlations and inflated positives."),
        spacer(),
        body("MicroNet addresses this in two ways. First, all abundance data undergoes Centered Log-Ratio (CLR) transformation, which maps compositions to a Euclidean space where standard statistical methods apply. Second, SPIEC-EASI inference uses sparse inverse covariance estimation, which explicitly conditions each pairwise relationship on all other taxa — filtering out indirect associations."),
        spacer(),
        callout("CLR(x\u1d62) = log(x\u1d62 / g(x))  where g(x) is the geometric mean of the composition. This removes the sum-to-one constraint and makes correlations interpretable."),
        spacer(),

        h2("2.3 Ecological Interaction Types"),
        body("The interaction matrix A inferred by gLV encodes five ecologically meaningful relationship types, classified by the signs of A\u1d62\u2c7c and A\u2c7c\u1d62:"),
        spacer(),
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [2000, 1200, 1200, 4960],
          rows: [
            new TableRow({ children: [
              headerCell("Interaction",  2000),
              headerCell("A\u1d62\u2c7c Sign", 1200),
              headerCell("A\u2c7c\u1d62 Sign", 1200),
              headerCell("Ecological Meaning", 4960),
            ]}),
            new TableRow({ children: [dataCell("Mutualism",   2000,true), dataCell("+",1200,true), dataCell("+",1200,true), dataCell("Both benefit — cross-feeding, metabolite exchange",4960,true)] }),
            new TableRow({ children: [dataCell("Competition", 2000),      dataCell("\u2212",1200), dataCell("\u2212",1200), dataCell("Both harmed — niche or resource overlap", 4960)] }),
            new TableRow({ children: [dataCell("Parasitism",  2000,true), dataCell("+",1200,true), dataCell("\u2212",1200,true), dataCell("One benefits, one is harmed", 4960,true)] }),
            new TableRow({ children: [dataCell("Commensalism",2000),      dataCell("+",1200), dataCell("0",1200), dataCell("One benefits, other unaffected", 4960)] }),
            new TableRow({ children: [dataCell("Amensalism",  2000,true), dataCell("\u2212",1200,true), dataCell("0",1200,true), dataCell("One harmed, other unaffected", 4960,true)] }),
          ],
        }),
        spacer(),

        // ─── 3. Datasets ────────────────────────────────────────────────────
        h1("3. Recommended Datasets"),
        body("MicroNet is validated on four public datasets spanning human gut, global soil, marine, and synthetic communities. All are freely accessible via NCBI SRA or EBI Metagenomics."),
        spacer(),

        h2("3.1 Primary Dataset: iHMP Phase 2"),
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [2400, 6960],
          rows: [
            new TableRow({ children: [headerCell("Property", 2400), headerCell("Details", 6960)] }),
            new TableRow({ children: [dataCell("Full name",   2400, true), dataCell("Integrative Human Microbiome Project Phase 2", 6960, true)] }),
            new TableRow({ children: [dataCell("BioProject",  2400),       dataCell("PRJNA398089", 6960)] }),
            new TableRow({ children: [dataCell("Access",      2400, true), dataCell("https://hmpdacc.org/ihmp/", 6960, true)] }),
            new TableRow({ children: [dataCell("Samples",     2400),       dataCell("~2,500 WGS shotgun metagenomes", 6960)] }),
            new TableRow({ children: [dataCell("Cohorts",     2400, true), dataCell("IBD (Crohn\u2019s, UC), Type 2 Diabetes, Pregnancy", 6960, true)] }),
            new TableRow({ children: [dataCell("Key feature", 2400),       dataCell("Longitudinal design: multiple time points per subject, enabling gLV temporal inference", 6960)] }),
          ],
        }),
        spacer(),
        body("The iHMP dataset is the primary training and validation dataset. Its longitudinal structure is critical for gLV inference, which requires estimating dx/dt from successive measurements. The disease cohorts provide biologically meaningful contrast — comparing healthy vs. disease microbiome networks tests the biological validity of inferred interactions."),
        spacer(),

        h2("3.2 Cross-Environment Comparison: Earth Microbiome Project"),
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [2400, 6960],
          rows: [
            new TableRow({ children: [headerCell("Property", 2400), headerCell("Details", 6960)] }),
            new TableRow({ children: [dataCell("Full name",   2400, true), dataCell("Earth Microbiome Project (EMP)", 6960, true)] }),
            new TableRow({ children: [dataCell("BioProject",  2400),       dataCell("PRJEB13870", 6960)] }),
            new TableRow({ children: [dataCell("Samples",     2400, true), dataCell("27,751 samples across 96 distinct biomes", 6960, true)] }),
            new TableRow({ children: [dataCell("Sequencing",  2400),       dataCell("16S rRNA V4 amplicon (DADA2 denoising)", 6960)] }),
            new TableRow({ children: [dataCell("Key feature", 2400, true), dataCell("Cross-biome comparison: test whether network topology is universal or environment-specific", 6960, true)] }),
          ],
        }),
        spacer(),

        h2("3.3 Benchmarking: CAMI2 Synthetic Communities"),
        body("CAMI2 provides simulated metagenomes with known ground-truth community composition, enabling rigorous validation of interaction inference methods. Because the true interaction strengths are specified by the simulation parameters, CAMI2 allows direct computation of inference accuracy (precision, recall, F1) — something impossible with real data where ground truth is unknown."),
        spacer(),
        bullet("URL: https://data.cami-challenge.org/"),
        bullet("Simulated from real genome databases with controlled complexity levels"),
        bullet("Used exclusively for benchmarking SPIEC-EASI vs. SparCC vs. gLV vs. GNN"),
        spacer(),

        h2("3.4 Marine Cross-Validation: TARA Oceans"),
        body("The TARA Oceans expedition sampled marine surface water across the globe, generating a uniquely diverse set of microbial communities with rich environmental metadata (temperature, salinity, nutrient concentrations). Using TARA Oceans as a held-out cross-validation set tests whether GNN-learned interaction embeddings generalize across radically different biomes."),
        spacer(),
        bullet("EBI Metagenomics Study ID: MGYS00002008"),
        bullet("~200 stations across Atlantic, Pacific, Indian, and Arctic Oceans"),
        bullet("Shotgun WGS with matched physical-chemical metadata"),
        spacer(),

        // ─── 4. Pipeline Architecture ──────────────────────────────────────
        h1("4. Pipeline Architecture"),
        body("The MicroNet pipeline is organized into five sequential phases, each implemented as a discrete software module. The full pipeline is orchestrated by Snakemake for reproducibility and parallel execution."),
        spacer(),

        h2("4.1 Phase 0: Data Acquisition and QC"),
        body("Raw paired-end FASTQ files are downloaded from NCBI SRA using fasterq-dump and pysradb. Quality control is performed with fastp, which trims Illumina adapters, removes low-quality bases (Q < 20), and filters reads shorter than 50 bp. QC metrics are aggregated into a MultiQC report for manual inspection before proceeding."),
        spacer(),
        bullet("Tool: fastp v0.23 — adapter trimming, quality filtering, length filtering"),
        bullet("Tool: fasterq-dump — parallel FASTQ retrieval from NCBI SRA"),
        bullet("Tool: MultiQC — aggregated QC report across all samples"),
        bullet("Checkpoint: samples with < 500,000 post-QC reads are flagged for exclusion"),
        spacer(),

        h2("4.2 Phase 1: Taxonomic and Functional Profiling"),
        body("Taxonomic composition is quantified at species level using MetaPhlAn4, which aligns reads against a curated database of clade-specific marker genes. Functional pathway abundance is estimated using HUMAnN3, which translates taxonomic profiles into KEGG pathway and gene family abundances. Both tools are run per sample; outputs are merged into community-level matrices."),
        spacer(),
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [2000, 2200, 2160, 2980],
          rows: [
            new TableRow({ children: [headerCell("Tool",2000), headerCell("Output",2200), headerCell("Resolution",2160), headerCell("Use in MicroNet",2980)] }),
            new TableRow({ children: [dataCell("MetaPhlAn4",2000,true), dataCell("Relative abundance",2200,true), dataCell("Species / strain",2160,true), dataCell("Node abundance profiles",2980,true)] }),
            new TableRow({ children: [dataCell("HUMAnN3",2000), dataCell("Pathway abundance",2200), dataCell("KEGG / MetaCyc",2160), dataCell("Node functional features",2980)] }),
            new TableRow({ children: [dataCell("DADA2 (16S)",2000,true), dataCell("ASV table",2200,true), dataCell("Sub-species",2160,true), dataCell("EMP amplicon data",2980,true)] }),
          ],
        }),
        spacer(),
        body("The merged species abundance matrix is then normalized using CLR transformation after filtering taxa present in fewer than 10% of samples or with mean relative abundance below 0.01%. This typically reduces the taxon count from thousands to 100-500 species of ecological relevance."),
        spacer(),

        h2("4.3 Phase 2: Interaction Inference"),
        h3("4.3.1 SPIEC-EASI Co-occurrence Network"),
        body("SPIEC-EASI (Sparse and Compositionally Robust Inference of Ecological Association Networks) constructs a sparse network by solving a graphical LASSO problem: for each taxon, it finds the minimal set of other taxa that explain its variance when conditioned on all others. Unlike pairwise correlations, this removes indirect associations — if A and B are both correlated with C, SPIEC-EASI correctly identifies that only two of the three relationships are direct."),
        spacer(),
        body("Network stability is assessed using the StARS (Stability Approach to Regularization Selection) criterion, which selects the sparsity parameter that produces stable edge selections across subsampled datasets. This produces a sparse, reproducible network with interpretable partial correlation edge weights."),
        spacer(),

        h3("4.3.2 Generalized Lotka-Volterra Inference"),
        body("The gLV model treats microbial community dynamics as a system of coupled differential equations: the per-capita growth rate of each taxon is a linear function of all other taxa\u2019s abundances. The interaction matrix A encodes these relationships, with A\u1d62\u2c7c > 0 indicating that taxon j promotes taxon i, and A\u1d62\u2c7c < 0 indicating inhibition."),
        spacer(),
        body("For longitudinal data, finite differences approximate dx/dt from successive abundance measurements. The system is then solved as a LASSO regression problem per taxon, with regularization enforcing sparsity (most taxa do not directly interact). The resulting interaction matrix is directed and signed — enabling classification of all five ecological interaction types."),
        spacer(),
        callout("Unlike co-occurrence methods which produce undirected, symmetric networks, gLV produces a directed graph: A\u1d62\u2c7c describes the effect of j on i, which may differ in sign and magnitude from A\u2c7c\u1d62."),
        spacer(),

        h3("4.3.3 Bayesian Network Structure Learning"),
        body("As a third inference method, Bayesian network structure learning (hill-climbing with BIC scoring) produces directed acyclic graphs (DAGs) representing conditional independence relationships. This approach is most useful for incorporating prior biological knowledge as informative priors — for example, known metabolic dependencies or phylogenetic constraints."),
        spacer(),

        h2("4.4 Phase 3: Graph Neural Network Training"),
        body("The VGAE (Variational Graph Autoencoder) learns rich latent representations of microbial taxa by encoding node features through graph convolution layers, then decoding by predicting edge existence from node embedding inner products. This architecture simultaneously achieves two goals: (1) link prediction, inferring high-confidence edges not captured by co-occurrence, and (2) representation learning, embedding taxa in a continuous latent space where similar ecological roles are geometrically proximate."),
        spacer(),

        h3("4.4.1 Node Features"),
        body("Each taxon node is described by a concatenated feature vector containing:"),
        bullet("CLR-transformed abundance profile across all samples (captures niche preferences)"),
        bullet("Functional gene content vector from HUMAnN3 (captures metabolic capabilities)"),
        bullet("Phylogenetic embedding — node2vec walk on PhyloT tree (captures evolutionary proximity)"),
        bullet("Genome-level features from MAGs: GC content, genome size, completeness (optional)"),
        spacer(),

        h3("4.4.2 Training Objective"),
        body("The model is trained to minimize the Evidence Lower Bound (ELBO): the sum of binary cross-entropy reconstruction loss (do predicted edges match observed co-occurrence edges?) and KL divergence (how far is the learned latent distribution from a standard Gaussian prior?). An auxiliary interaction type classification head is trained jointly using gLV-labeled edges as supervision, improving the semantic content of node embeddings."),
        spacer(),

        h3("4.4.3 Link Prediction"),
        body("After training, the full N\u00d7N edge probability matrix is computed from node embedding inner products. Edges with probability > 0.7 that are absent from the SPIEC-EASI network are added as GNN-predicted interactions, substantially increasing network coverage for rare or low-abundance taxa."),
        spacer(),

        h2("4.5 Phase 4: Network Topology Analysis"),
        body("The consensus network (SPIEC-EASI edges annotated with gLV types, plus high-confidence GNN predictions) is analyzed using NetworkX and iGraph for ecological interpretation."),
        spacer(),
        bullet("Keystone taxa identification: composite centrality score from betweenness, eigenvector, closeness, and hub scores"),
        bullet("Ecological guild detection: Leiden community detection reveals functional clusters of mutually interacting taxa"),
        bullet("Robustness simulation: sequential taxon removal under targeted (betweenness-first) and random strategies, tracking community collapse"),
        bullet("Small-world testing: comparison of clustering coefficient and path length to random graphs (Watts-Strogatz sigma)"),
        bullet("Scale-free testing: power-law fit to degree distribution (scale-free networks have hubs that disproportionately stabilize the community)"),
        spacer(),

        h2("4.6 Phase 5: Visualization Dashboard"),
        body("A Streamlit web dashboard provides interactive exploration of all pipeline outputs without requiring programming knowledge. The dashboard renders a force-directed network graph using Plotly with interactive node hovering, edge filtering by weight and interaction type, layout algorithm selection, and taxon drilldown. Additional tabs display keystone taxon rankings, guild compositions, robustness curves, and topology summary statistics."),
        spacer(),

        // ─── 5. Technology Stack ────────────────────────────────────────────
        h1("5. Technology Stack"),
        spacer(),
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [2400, 2960, 3980],
          rows: [
            new TableRow({ children: [headerCell("Layer",2400), headerCell("Tools",2960), headerCell("Purpose",3980)] }),
            new TableRow({ children: [dataCell("Profiling",      2400,true), dataCell("MetaPhlAn4, HUMAnN3, DADA2",2960,true), dataCell("Taxonomic & functional quantification",3980,true)] }),
            new TableRow({ children: [dataCell("Co-occurrence",  2400),      dataCell("SPIEC-EASI, SparCC, FlashWeave",2960),      dataCell("Sparse correlation network baseline",3980)] }),
            new TableRow({ children: [dataCell("Dynamics",       2400,true), dataCell("LASSO/Ridge regression (sklearn)",2960,true),dataCell("gLV interaction matrix inference",3980,true)] }),
            new TableRow({ children: [dataCell("Bayesian",       2400),      dataCell("PyMC, pgmpy, Stan",2960),                   dataCell("Probabilistic interaction inference",3980)] }),
            new TableRow({ children: [dataCell("GNN",            2400,true), dataCell("PyTorch Geometric, DGL",2960,true),          dataCell("Link prediction, representation learning",3980,true)] }),
            new TableRow({ children: [dataCell("Network",        2400),      dataCell("NetworkX, iGraph, leidenalg",2960),          dataCell("Topology analysis, community detection",3980)] }),
            new TableRow({ children: [dataCell("Metabolic",      2400,true), dataCell("MICOM, COBRApy",2960,true),                  dataCell("Interaction validation via flux modeling",3980,true)] }),
            new TableRow({ children: [dataCell("Visualization",  2400),      dataCell("Streamlit, Plotly, Gephi",2960),             dataCell("Interactive dashboard & publication figures",3980)] }),
            new TableRow({ children: [dataCell("Pipeline",       2400,true), dataCell("Snakemake, conda",2960,true),                dataCell("Reproducible workflow orchestration",3980,true)] }),
          ],
        }),
        spacer(),

        // ─── 6. Novelty and Research Contributions ─────────────────────────
        h1("6. Novelty and Research Contributions"),
        body("MicroNet makes the following original contributions relative to existing tools and published methods:"),
        spacer(),

        h2("6.1 Multi-method Ensemble"),
        body("Existing tools (SPIEC-EASI, SparCC, FlashWeave) each implement a single inference strategy. MicroNet is, to our knowledge, the first systematic ensemble that combines co-occurrence, dynamical (gLV), and representation learning (GNN) approaches within a unified pipeline — with explicit benchmarking of each method against ground truth (CAMI2) and against each other."),
        spacer(),

        h2("6.2 Supervised Interaction Type Labeling"),
        body("Most existing network tools produce unsigned or weakly signed edges without ecological interpretation. MicroNet introduces the first GNN architecture with a jointly trained interaction type classification head — using gLV-inferred labels as weak supervision. This means node embeddings encode not just co-occurrence patterns, but ecological role."),
        spacer(),

        h2("6.3 Dynamic Network Reconstruction"),
        body("By exploiting the longitudinal structure of iHMP data, MicroNet can track how microbial ecological networks rewire during disease progression, treatment, or environmental perturbation. Using EvolveGCN or temporal graph attention networks on the longitudinal subsets extends the framework to detect dynamic interaction changes."),
        spacer(),

        h2("6.4 Robustness-Informed Keystone Identification"),
        body("Existing keystone taxa analyses use a single centrality metric (usually betweenness). MicroNet introduces a composite keystone score from six centrality metrics, validated by robustness simulation: taxa ranked highly by the composite score should trigger faster community collapse when removed. This provides an empirical, rather than purely structural, definition of ecological keystoneness."),
        spacer(),

        // ─── 7. Validation Strategy ─────────────────────────────────────────
        h1("7. Validation Strategy"),
        spacer(),

        h2("7.1 Synthetic Community Benchmarking"),
        body("Using CAMI2 synthetic metagenomes with known community composition, we validate each inference method by:"),
        bullet("Precision and recall of inferred edges relative to known species co-occurrence"),
        bullet("F1 score for interaction type classification (mutualism vs. competition etc.) against simulation parameters"),
        bullet("AUC-ROC for binary edge prediction"),
        bullet("Spearman correlation between inferred A matrix and true interaction coefficients"),
        spacer(),

        h2("7.2 Cross-Dataset Generalization"),
        body("GNN models trained on iHMP gut data are tested zero-shot on TARA Oceans marine data. Better-than-random link prediction AUC on the held-out dataset indicates that learned embeddings capture biologically general interaction patterns rather than dataset-specific artifacts."),
        spacer(),

        h2("7.3 Biological Validation"),
        body("Inferred cooperative interactions (mutualism, commensalism) are validated against published cross-feeding literature. For example, Faecalibacterium prausnitzii and butyrate producers are known to cooperate in the healthy gut microbiome; their positive interaction should appear in the network. Predicted keystone taxa are cross-referenced against published experimental depletion studies."),
        spacer(),

        h2("7.4 Stability Analysis"),
        body("The gLV model predicts community stability from the eigenvalues of the Jacobian at equilibrium. Communities with all negative real eigenvalues are locally stable. We test whether inferred interaction matrices predict stability concordant with observed community temporal stability in longitudinal iHMP samples."),
        spacer(),

        // ─── 8. Expected Outputs ─────────────────────────────────────────────
        h1("8. Expected Outputs"),
        spacer(),
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [3000, 6360],
          rows: [
            new TableRow({ children: [headerCell("Output", 3000), headerCell("Description", 6360)] }),
            new TableRow({ children: [dataCell("Ecological network graph",   3000,true), dataCell("Signed, weighted, directed graph (N\u00d7N, ~100\u2013500 taxa). Nodes = species; edges = inferred interactions with type and confidence score.",6360,true)] }),
            new TableRow({ children: [dataCell("Keystone taxa ranking",       3000),      dataCell("Ranked list of taxa by composite centrality score with predicted impact of removal.",6360)] }),
            new TableRow({ children: [dataCell("Ecological guild table",     3000,true), dataCell("Community partitioning into functional guilds with member taxa and characteristic metabolic functions.",6360,true)] }),
            new TableRow({ children: [dataCell("gLV interaction matrix",     3000),      dataCell("Directed A matrix with per-pair interaction coefficients, confidence intervals, and ecological type labels.",6360)] }),
            new TableRow({ children: [dataCell("GNN node embeddings",        3000,true), dataCell("64-dimensional latent space per taxon. Visualizable by UMAP — similar ecological roles cluster together.",6360,true)] }),
            new TableRow({ children: [dataCell("Robustness curves",          3000),      dataCell("Community collapse curves under targeted vs. random taxon removal, with collapse thresholds.",6360)] }),
            new TableRow({ children: [dataCell("Interactive dashboard",      3000,true), dataCell("Streamlit web application for network exploration, filtering, and drilldown without code.",6360,true)] }),
            new TableRow({ children: [dataCell("Benchmarking report",        3000),      dataCell("Precision/recall/AUC comparison of SPIEC-EASI, SparCC, gLV, and GNN on CAMI2 ground truth.",6360)] }),
          ],
        }),
        spacer(),

        // ─── 9. Project Timeline ────────────────────────────────────────────
        h1("9. Suggested Project Timeline"),
        spacer(),
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [1400, 2200, 5760],
          rows: [
            new TableRow({ children: [headerCell("Week", 1400), headerCell("Phase", 2200), headerCell("Deliverable", 5760)] }),
            new TableRow({ children: [dataCell("1\u20132",   1400,true), dataCell("Data & Setup",     2200,true), dataCell("Conda environment configured; 100 iHMP samples downloaded, QC'd",  5760,true)] }),
            new TableRow({ children: [dataCell("3\u20134",   1400),      dataCell("Profiling",         2200),      dataCell("MetaPhlAn4 + HUMAnN3 run; CLR matrix generated",                   5760)] }),
            new TableRow({ children: [dataCell("5\u20136",   1400,true), dataCell("Co-occurrence",     2200,true), dataCell("SPIEC-EASI and SparCC networks; network statistics",               5760,true)] }),
            new TableRow({ children: [dataCell("7\u20138",   1400),      dataCell("gLV inference",     2200),      dataCell("Interaction matrix A fitted; ecological types classified",          5760)] }),
            new TableRow({ children: [dataCell("9\u201310",  1400,true), dataCell("GNN training",      2200,true), dataCell("VGAE trained; link prediction AUC > 0.80 on validation set",      5760,true)] }),
            new TableRow({ children: [dataCell("11\u201312", 1400),      dataCell("Analysis",          2200),      dataCell("Keystone taxa, guilds, robustness, small-world testing",           5760)] }),
            new TableRow({ children: [dataCell("13\u201314", 1400,true), dataCell("Benchmarking",      2200,true), dataCell("CAMI2 ground-truth comparison; method ranking table",             5760,true)] }),
            new TableRow({ children: [dataCell("15\u201316", 1400),      dataCell("Dashboard & Docs",  2200),      dataCell("Streamlit dashboard deployed; README and report finalized",        5760)] }),
          ],
        }),
        spacer(),

        // ─── 10. References ─────────────────────────────────────────────────
        h1("10. Key References"),
        body("The following works form the methodological foundation of MicroNet:"),
        spacer(),
        bullet("Kurtz et al. (2015). Sparse and compositionally robust inference of microbial ecological networks. PLoS Computational Biology."),
        bullet("Faust & Raes (2012). Microbial interactions: from networks to models. Nature Reviews Microbiology."),
        bullet("Stein et al. (2013). Ecological modeling from time-series inference: insight into dynamics and stability of intestinal microbiota. PLoS Computational Biology."),
        bullet("Kipf & Welling (2016). Variational Graph Autoencoders. arXiv:1611.07308."),
        bullet("Franzosa et al. (2018). Species-level functional profiling of metagenomes and metatranscriptomes. Nature Methods (HUMAnN3)."),
        bullet("Beghini et al. (2021). Integrating taxonomic, functional, and strain-level profiling of diverse microbial communities with bioBakery 3. eLife (MetaPhlAn4)."),
        bullet("Lloyd-Price et al. (2019). Multi-omics of the gut microbial ecosystem in inflammatory bowel diseases. Nature (iHMP IBD)."),
        bullet("Thompson et al. (2017). A communal catalogue reveals Earth\u2019s multiscale microbial diversity. Nature (EMP)."),
        bullet("Sunagawa et al. (2015). Structure and function of the global ocean microbiome. Science (TARA Oceans)."),
        spacer(),

        // ─── Footer note ───────────────────────────────────────────────────
        callout("This document describes MicroNet v1.0. The full codebase, Snakemake workflow, and environment specifications are available in the project repository. For questions on methodology, refer to Section 4 (Pipeline Architecture) and Section 7 (Validation Strategy)."),
      ],
    },
  ],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("/mnt/user-data/outputs/MicroNet_Project_Document.docx", buf);
  console.log("Document written successfully.");
});
