#!/usr/bin/env python3
"""
glv_inference.py — Infer microbial interaction matrix A from the generalized
                   Lotka-Volterra (gLV) model using LASSO-regularized regression.

Model:
    dx_i/dt = x_i * (r_i + sum_j A_ij * x_j)

Where:
    x_i   = abundance of taxon i
    r_i   = intrinsic growth rate of taxon i
    A_ij  = effect of taxon j on taxon i
             A_ij > 0  → j promotes i  (mutualism / cooperation)
             A_ij < 0  → j inhibits i  (competition / amensalism)
             A_ii < 0  → self-regulation (carrying capacity)

For time-series data, we can linearize:
    (1/x_i) * dx_i/dt = r_i + A_i1*x_1 + ... + A_iN*x_N

And solve as a regression problem (LASSO for sparsity).

For cross-sectional data (no time series), use the pseudo-steady-state approach.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ── Core inference ─────────────────────────────────────────────────────────

def compute_derivatives(X: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Estimate dx/dt from time-series abundance data using finite differences.

    Parameters
    ----------
    X     : (T, N) array — abundance of N taxa at T time points
    times : (T,) array — time points in days

    Returns
    -------
    dX_dt : (T-1, N) array — derivative estimates at midpoints
    X_mid : (T-1, N) array — abundance at midpoints (for regression)
    """
    dX_dt = np.diff(X, axis=0) / np.diff(times)[:, None]
    X_mid = (X[:-1] + X[1:]) / 2
    return dX_dt, X_mid


def fit_glv_lasso(X: np.ndarray,
                  dX_dt: np.ndarray,
                  alpha_range: np.ndarray = None,
                  cv: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit gLV interaction matrix using LASSO regression per taxon.

    For each taxon i, solve:
        (1/x_i) * dx_i/dt = r_i + A_i @ X

    Returns
    -------
    A : (N, N) interaction matrix
    r : (N,) intrinsic growth rates
    """
    N = X.shape[1]

    # FIX C1: caller already aligns X and dX_dt to the same shape (both T-1, N).
    # Never drop another row here — that caused a shape mismatch and silently
    # corrupted every interaction coefficient.
    assert X.shape == dX_dt.shape, (
        f"X {X.shape} and dX_dt {dX_dt.shape} must be the same shape. "
        "Ensure compute_derivatives() output is passed directly."
    )

    # Avoid division by zero
    X_safe = np.where(X > 1e-10, X, 1e-10)

    # Per-capita growth rate: (1/x_i) * dx_i/dt
    percap_growth = dX_dt / X_safe   # Both are (T-1, N) — shapes match

    A = np.zeros((N, N))
    r = np.zeros(N)

    if alpha_range is None:
        alpha_range = np.logspace(-4, 1, 50)

    for i in range(N):
        y = percap_growth[:, i]
        model = LassoCV(alphas=alpha_range, cv=cv, max_iter=10000, fit_intercept=True)
        model.fit(X, y)

        r[i] = model.intercept_
        A[i, :] = model.coef_

    return A, r


def fit_glv_ridge(X: np.ndarray, dX_dt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit gLV using Ridge regression — useful when N > samples (overdetermined).
    More stable than LASSO but less sparse.
    """
    N = X.shape[1]
    # FIX M2: guard against misaligned inputs (same root cause as C1)
    assert X.shape == dX_dt.shape, (
        f"X {X.shape} and dX_dt {dX_dt.shape} must be the same shape."
    )
    X_safe = np.where(X > 1e-10, X, 1e-10)
    percap_growth = dX_dt / X_safe

    A = np.zeros((N, N))
    r = np.zeros(N)

    model = RidgeCV(alphas=np.logspace(-3, 3, 50), fit_intercept=True)
    for i in range(N):
        model.fit(X, percap_growth[:, i])
        r[i] = model.intercept_
        A[i, :] = model.coef_

    return A, r


def fit_glv_pss(X: np.ndarray,
                alpha_range: np.ndarray = None,
                method: str = "lasso") -> tuple[np.ndarray, np.ndarray]:
    """
    Pseudo-steady-state (PSS) interaction inference for cross-sectional data.

    Assumes samples are near equilibrium: dx_i/dt ≈ 0, so r_i ≈ -Σ_j A_ij * x_j.
    For each taxon i, regress x_i against all other taxa x_j (LIMITS-style):
        x_i ≈ intercept + Σ_{j≠i} A_ij * x_j

    This is correct because at steady state the intercept absorbs -r_i/A_ii
    and the coefficients recover the off-diagonal interaction strengths.

    Reference: Fisher & Mehta (2014), PLOS ONE — LIMITS algorithm.
    """
    N, S = X.shape[1], X.shape[0]
    A = np.zeros((N, N))
    r = np.zeros(N)

    if alpha_range is None:
        alpha_range = np.logspace(-4, 1, 50)

    for i in range(N):
        # Target: abundance of taxon i across all samples
        y = X[:, i]
        # Predictors: abundances of all OTHER taxa
        X_others = np.delete(X, i, axis=1)

        if method == "lasso":
            model = LassoCV(alphas=alpha_range, cv=min(5, S), max_iter=10000,
                            fit_intercept=True)
        else:
            model = RidgeCV(alphas=np.logspace(-3, 3, 50), fit_intercept=True)

        model.fit(X_others, y)
        r[i] = model.intercept_
        # Map coefficients back into full N-length row, skipping diagonal
        coefs = model.coef_
        A[i, :i]   = coefs[:i]
        A[i, i+1:] = coefs[i:]
        # Self-regulation: diagonal estimated as negative of mean coefficient magnitude
        A[i, i] = -np.abs(coefs).mean()

    return A, r


def classify_interactions(A: np.ndarray,
                           threshold: float = 0.01) -> pd.DataFrame:
    """
    Classify pairwise interactions based on sign of A_ij and A_ji.

    Ecological interaction types (i affected by j):
        A_ij > 0, A_ji > 0  → Mutualism (+/+)
        A_ij < 0, A_ji < 0  → Competition (-/-)
        A_ij > 0, A_ji < 0  → Parasitism / Predation (+/-)
        A_ij < 0, A_ji > 0  → Parasitism / Predation (-/+)
        A_ij > 0, A_ji ≈ 0  → Commensalism (+/0)
        A_ij < 0, A_ji ≈ 0  → Amensalism (-/0)
    """
    N = A.shape[0]
    records = []

    for i in range(N):
        for j in range(i + 1, N):
            a_ij = A[i, j]
            a_ji = A[j, i]

            if abs(a_ij) < threshold and abs(a_ji) < threshold:
                continue  # No significant interaction
            strength = (abs(a_ij) + abs(a_ji)) / 2
            if strength < threshold:
                # FIX m5: one side may barely exceed threshold while the average
                # is still below — skip to avoid classifying noise as interactions.
                continue

            s_ij = "+" if a_ij > threshold else ("-" if a_ij < -threshold else "0")
            s_ji = "+" if a_ji > threshold else ("-" if a_ji < -threshold else "0")
            signature = f"{s_ij}/{s_ji}"

            itype = {
                "+/+": "Mutualism",
                "-/-": "Competition",
                "+/-": "Parasitism",
                "-/+": "Parasitism",
                "+/0": "Commensalism",
                "0/+": "Commensalism",
                "-/0": "Amensalism",
                "0/-": "Amensalism",
            }.get(signature, "Unknown")

            records.append({
                "taxon_i": i, "taxon_j": j,
                "A_ij": a_ij, "A_ji": a_ji,
                "signature": signature,
                "interaction_type": itype,
                "strength": (abs(a_ij) + abs(a_ji)) / 2,
            })

    return pd.DataFrame(records)


# ── Stability analysis ──────────────────────────────────────────────────────

def community_stability(A: np.ndarray, x_eq: np.ndarray) -> dict:
    """
    Assess local stability of the community at equilibrium x_eq.
    Computes Jacobian eigenvalues — all negative real parts = stable.

    Returns
    -------
    dict with eigenvalues, stability flag, and resilience (max real eigenvalue)
    """
    # Jacobian J_ij = x_i * A_ij (for i != j), J_ii = r_i + 2*A_ii*x_i + sum_{j!=i} A_ij*x_j
    N = len(x_eq)
    J = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            J[i, j] = x_eq[i] * A[i, j]

    eigenvalues = np.linalg.eigvals(J)
    max_real = np.max(np.real(eigenvalues))

    return {
        "eigenvalues": eigenvalues,
        "is_stable": max_real < 0,
        "resilience": -max_real,          # Higher = faster return to equilibrium
        "reactivity": np.max(np.real(     # Transient amplification before settling
            np.linalg.eigvals((J + J.T) / 2)
        )),
    }


# ── Visualization ───────────────────────────────────────────────────────────

def plot_interaction_matrix(A: np.ndarray,
                             taxa_names: list[str],
                             output_path: str = None) -> None:
    """Heatmap of the interaction matrix with diverging color scale."""
    fig, ax = plt.subplots(figsize=(12, 10))
    vmax = np.percentile(np.abs(A), 95)
    sns.heatmap(A,
                xticklabels=taxa_names, yticklabels=taxa_names,
                cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                ax=ax, square=True, linewidths=0.3)
    ax.set_title("gLV Interaction Matrix (A_ij: effect of j on i)", fontsize=14)
    ax.set_xlabel("Source taxon (j)")
    ax.set_ylabel("Target taxon (i)")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="gLV interaction inference")
    parser.add_argument("--abundance", required=True, help="CLR abundance matrix TSV")
    parser.add_argument("--metadata", required=True, help="Sample metadata TSV (must have 'time_point' column)")
    parser.add_argument("--method", choices=["lasso", "ridge"], default="lasso")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Interaction coefficient threshold for classification")
    parser.add_argument("--outdir", default="../../results/inference")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    clr = pd.read_csv(args.abundance, sep="\t", index_col=0)
    meta = pd.read_csv(args.metadata, sep="\t", index_col=0)

    taxa = clr.columns.tolist()
    print(f"Loaded: {clr.shape[0]} samples × {clr.shape[1]} taxa")

    # Check for time-series structure
    if "time_point" in meta.columns and "subject_id" in meta.columns:
        print("Time-series data detected — using temporal gLV inference")
        # Group by subject, stack time points
        all_X, all_dX = [], []
        for subj, grp in meta.groupby("subject_id"):
            grp = grp.sort_values("time_point")
            X = clr.loc[grp.index].values
            times = grp["time_point"].values.astype(float)
            dX, X_mid = compute_derivatives(X, times)
            all_X.append(X_mid)
            all_dX.append(dX)
        X_all = np.vstack(all_X)
        dX_all = np.vstack(all_dX)
    else:
        # FIX C1 (v2): The previous fix set dX_all=zeros, which made every
        # per-capita growth rate zero → LASSO learned A=0 for all taxa.
        # Correct pseudo-steady-state formulation: at equilibrium, r_i = -A_i @ x,
        # so regress each taxon's abundance against all others' abundances.
        # This is handled in a dedicated path rather than going through fit_glv_lasso.
        print("Cross-sectional data — using pseudo-steady-state (LIMITS-style) inference")
        X_all = clr.values   # (S, N) — will be used directly below
        dX_all = None        # Sentinel: triggers PSS branch in fit function

    # Fit model
    print(f"Fitting gLV via {args.method.upper()} ...")
    if dX_all is None:
        # Cross-sectional: use pseudo-steady-state LIMITS-style regression
        A, r = fit_glv_pss(X_all, alpha_range=None if args.method == "lasso" else None,
                           method=args.method)
    elif args.method == "lasso":
        A, r = fit_glv_lasso(X_all, dX_all)
    else:
        A, r = fit_glv_ridge(X_all, dX_all)

    # Save interaction matrix
    A_df = pd.DataFrame(A, index=taxa, columns=taxa)
    A_df.to_csv(outdir / "glv_interaction_matrix.tsv", sep="\t")
    print(f"Interaction matrix saved → {outdir}/glv_interaction_matrix.tsv")

    # Classify interactions
    interactions = classify_interactions(A, threshold=args.threshold)
    interactions["taxon_i_name"] = interactions["taxon_i"].map(lambda i: taxa[i])
    interactions["taxon_j_name"] = interactions["taxon_j"].map(lambda i: taxa[i])
    interactions.to_csv(outdir / "classified_interactions.tsv", sep="\t", index=False)
    print(f"Classified {len(interactions)} interactions")
    print(interactions["interaction_type"].value_counts())

    # Stability
    # FIX M1: CLR values are log-ratios and can be negative — abs(CLR) has no
    # biological meaning as an abundance.  Convert CLR back to proportions first.
    clr_means   = clr.mean().values
    raw_props   = np.exp(clr_means)                        # undo the log
    x_eq        = raw_props / raw_props.sum()              # renormalize to proportions
    stab = community_stability(A, x_eq)
    print(f"\nCommunity stability: {'STABLE' if stab['is_stable'] else 'UNSTABLE'}")
    print(f"  Resilience:  {stab['resilience']:.4f}")
    print(f"  Reactivity:  {stab['reactivity']:.4f}")

    # FIX M6: passing the full (N,N) A matrix with only 30 labels causes
    # sns.heatmap to misalign every tick label past the 30th row/column.
    # Slice the matrix to match the displayed taxa.
    n_show = min(30, len(taxa))
    plot_interaction_matrix(A[:n_show, :n_show], taxa[:n_show],
                            output_path=str(outdir / "interaction_matrix_heatmap.png"))


if __name__ == "__main__":
    main()
