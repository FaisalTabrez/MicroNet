#!/usr/bin/env python3
"""
clr_normalize.py — Centered Log-Ratio normalization for compositional metagenomics data.

CLR is preferred over simple relative abundance because it removes the
compositionality constraint (samples sum to 1), making distances Euclidean
and correlations interpretable.

Reference: Aitchison, J. (1986). The Statistical Analysis of Compositional Data.
"""

import numpy as np
import pandas as pd
# CLR uses log-space arithmetic: log(x) - mean(log(x)) = log(x / geometric_mean(x))
# scipy.stats.gmean is not needed — removed to avoid importing dead code.


def clr_transform(df: pd.DataFrame, pseudocount: float = 1e-6) -> pd.DataFrame:
    """
    Apply Centered Log-Ratio transformation.

    CLR(x_i) = log(x_i / g(x))
    where g(x) is the geometric mean of the composition.

    Parameters
    ----------
    df : samples × taxa abundance matrix (relative abundances or counts)
    pseudocount : added before log to handle zeros

    Returns
    -------
    CLR-transformed matrix, same shape as input
    """
    X = df.values.astype(float)
    X = X + pseudocount                        # Replace zeros
    X = X / X.sum(axis=1, keepdims=True)       # Re-normalize to proportions
    log_X = np.log(X)
    clr_X = log_X - log_X.mean(axis=1, keepdims=True)  # Subtract row geometric mean
    return pd.DataFrame(clr_X, index=df.index, columns=df.columns)


def filter_low_prevalence(df: pd.DataFrame,
                           min_prevalence: float = 0.1,
                           min_abundance: float = 1e-4) -> pd.DataFrame:
    """
    Remove taxa present in fewer than min_prevalence fraction of samples,
    or with mean relative abundance below min_abundance.

    Reduces noise and speeds up downstream inference.
    """
    # Prevalence filter
    prevalence = (df > 0).mean(axis=0)
    keep_prev = prevalence[prevalence >= min_prevalence].index

    # Abundance filter
    mean_abund = df.mean(axis=0)
    keep_abund = mean_abund[mean_abund >= min_abundance].index

    keep = keep_prev.intersection(keep_abund)
    filtered = df[keep]
    print(f"Filtered: {df.shape[1]} → {filtered.shape[1]} taxa "
          f"(removed {df.shape[1] - filtered.shape[1]} low-prevalence/low-abundance)")
    return filtered


def load_metaphlan_table(path: str) -> pd.DataFrame:
    """
    Load merged MetaPhlAn4 output table.
    Keeps only species-level rows (s__), transposes to samples × taxa.
    """
    df = pd.read_csv(path, sep="\t", skiprows=1, index_col=0)

    # Keep species level only
    species_mask = df.index.str.contains(r"\|s__") & ~df.index.str.contains(r"\|t__")
    df = df[species_mask]

    # Clean taxon names
    df.index = df.index.str.extract(r"s__(.+)$")[0].str.replace("_", " ")

    # Transpose: samples × taxa
    df = df.T
    df.index.name = "sample_id"

    return df


if __name__ == "__main__":
    import sys

    input_path  = snakemake.input[0]   # noqa: F821 (Snakemake injects this)
    output_path = snakemake.output[0]  # noqa: F821

    print(f"Loading MetaPhlAn table: {input_path}")
    raw = load_metaphlan_table(input_path)
    print(f"Raw shape: {raw.shape} (samples × taxa)")

    filtered = filter_low_prevalence(raw, min_prevalence=0.1, min_abundance=1e-4)
    clr = clr_transform(filtered)

    clr.to_csv(output_path, sep="\t")
    print(f"CLR matrix saved → {output_path}  shape: {clr.shape}")
