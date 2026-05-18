#!/usr/bin/env python3
"""
download_data.py — Fetch metagenome datasets from SRA / EBI for MicroNet pipeline.

Usage:
    python download_data.py --dataset hmp --n-samples 100 --outdir ../../data/raw
    python download_data.py --dataset emp --n-samples 200 --outdir ../../data/raw
    python download_data.py --dataset cami --outdir ../../data/raw
"""

import argparse
import logging
import os
import subprocess
from pathlib import Path

import pandas as pd
from pysradb import SRAweb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Dataset registry ────────────────────────────────────────────────────────
DATASETS = {
    "hmp": {
        "bioproject": "PRJNA398089",
        "description": "iHMP — Integrative Human Microbiome Project Phase 2",
        "metadata_url": "https://hmpdacc.org/ihmp/",
        "notes": "Longitudinal IBD/T2D/pregnancy cohorts, WGS shotgun",
    },
    "emp": {
        "bioproject": "PRJEB13870",
        "description": "Earth Microbiome Project — 16S V4 amplicon, 96 biomes",
        "metadata_url": "https://earthmicrobiome.org/",
        "notes": "Cross-environment comparison dataset",
    },
    "cami": {
        "description": "CAMI2 synthetic benchmark communities",
        "direct_url": "https://data.cami-challenge.org/participate",
        "notes": "Ground-truth known community — use for benchmarking only",
    },
}


def fetch_sra_metadata(bioproject: str, n_samples: int) -> pd.DataFrame:
    """Retrieve sample metadata from SRA for a given BioProject."""
    log.info(f"Fetching SRA metadata for {bioproject} ...")
    db = SRAweb()
    df = db.search_by_accession(bioproject)
    df = df.head(n_samples)
    log.info(f"Found {len(df)} samples")
    return df


def download_fastq(accession: str, outdir: Path, threads: int = 4) -> None:
    """Download paired FASTQ files using fasterq-dump."""
    outdir.mkdir(parents=True, exist_ok=True)
    r1_gz  = outdir / f"{accession}_1.fastq.gz"
    r1_raw = outdir / f"{accession}_1.fastq"
    # FIX C7: fasterq-dump produces uncompressed .fastq; pigz creates .gz afterward.
    # If pigz fails the .fastq is left on disk but the .gz check misses it,
    # triggering a redundant full re-download. Check for either form.
    if r1_gz.exists() or r1_raw.exists():
        log.info(f"  {accession} already downloaded, skipping.")
        return

    log.info(f"  Downloading {accession} ...")
    cmd = [
        "fasterq-dump", accession,
        "--outdir", str(outdir),
        "--threads", str(threads),
        "--split-files",
        "--skip-technical",
    ]
    subprocess.run(cmd, check=True)

    # Compress
    subprocess.run(["pigz", "-p", str(threads),
                    str(outdir / f"{accession}_1.fastq"),
                    str(outdir / f"{accession}_2.fastq")], check=True)


def run_qc(accession: str, rawdir: Path, qcdir: Path, threads: int = 4) -> None:
    """Trim adapters and low-quality bases with fastp."""
    qcdir.mkdir(parents=True, exist_ok=True)
    r1_in  = rawdir / f"{accession}_1.fastq.gz"
    r2_in  = rawdir / f"{accession}_2.fastq.gz"
    r1_out = qcdir  / f"{accession}_1.fastq.gz"
    r2_out = qcdir  / f"{accession}_2.fastq.gz"

    if r1_out.exists():
        log.info(f"  {accession} QC already done, skipping.")
        return

    log.info(f"  QC trimming {accession} ...")
    cmd = [
        "fastp",
        "-i", str(r1_in), "-I", str(r2_in),
        "-o", str(r1_out), "-O", str(r2_out),
        "--thread", str(threads),
        "--detect_adapter_for_pe",
        "--length_required", "50",
        "--json", str(qcdir / f"{accession}_fastp.json"),
        "--html", str(qcdir / f"{accession}_fastp.html"),
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Download metagenome datasets for MicroNet")
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Number of samples to download (default: 50)")
    parser.add_argument("--outdir", default="../../data/raw",
                        help="Output directory for raw reads")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--qc", action="store_true",
                        help="Run fastp QC after download")
    args = parser.parse_args()

    ds = DATASETS[args.dataset]
    log.info(f"Dataset: {ds['description']}")

    rawdir = Path(args.outdir) / args.dataset
    qcdir  = Path(args.outdir).parent / "qc" / args.dataset

    if "bioproject" not in ds:
        log.warning(f"No SRA BioProject for {args.dataset}. Visit: {ds.get('direct_url', '')}")
        return

    meta = fetch_sra_metadata(ds["bioproject"], args.n_samples)
    meta_path = rawdir / "metadata.tsv"
    rawdir.mkdir(parents=True, exist_ok=True)
    meta.to_csv(meta_path, sep="\t", index=False)
    log.info(f"Metadata saved → {meta_path}")

    # FIX m6: different pysradb versions return different column names
    # (run_accession, Run, SRR). Fuzzy-match so the script doesn't silently
    # download nothing when the name doesn't match exactly.
    acc_col = next(
        (c for c in meta.columns if c.lower() in ("run_accession", "run", "srr")),
        None,
    )
    if acc_col is None:
        log.error(
            f"Cannot find accession column. Available columns: {meta.columns.tolist()}"
        )
        return
    accessions = meta[acc_col].tolist()
    for acc in accessions:
        try:
            download_fastq(acc, rawdir, threads=args.threads)
            if args.qc:
                run_qc(acc, rawdir, qcdir, threads=args.threads)
        except subprocess.CalledProcessError as e:
            log.error(f"Failed on {acc}: {e}")

    log.info("Done.")


if __name__ == "__main__":
    main()
