#!/usr/bin/env python3
"""
topology.py — Network topology analysis for microbial ecological networks.

Computes:
    - Centrality metrics (identify keystone taxa)
    - Community detection (ecological guilds)
    - Robustness simulation (taxon removal cascades)
    - Small-world / scale-free property testing
    - Interaction type distribution
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
import argparse
from scipy import stats


# ── Network construction ──────────────────────────────────────────────────────

def build_network(adj_path: str,
                  interactions_path: str = None,
                  gnn_prob_path: str = None,
                  gnn_threshold: float = 0.7) -> nx.Graph:
    """
    Build a NetworkX graph from adjacency matrix.
    Optionally overlay gLV interaction type labels and GNN-predicted edges.
    """
    adj_df = pd.read_csv(adj_path, sep="\t", index_col=0)
    taxa = adj_df.index.tolist()

    G = nx.Graph()
    G.add_nodes_from(taxa)

    # Add edges from SPIEC-EASI adjacency
    for i, t1 in enumerate(taxa):
        for j, t2 in enumerate(taxa[i+1:], i+1):
            w = adj_df.iloc[i, j]
            if w != 0:
                G.add_edge(t1, t2,
                           weight=abs(w),
                           sign="positive" if w > 0 else "negative",
                           source="spieceasi",
                           interaction_type="Unknown")

    # Annotate with gLV interaction types
    if interactions_path:
        idf = pd.read_csv(interactions_path, sep="\t")
        for _, row in idf.iterrows():
            t1 = row.get("taxon_i_name")
            t2 = row.get("taxon_j_name")
            if G.has_edge(t1, t2):
                G[t1][t2]["interaction_type"] = row["interaction_type"]
                G[t1][t2]["A_ij"] = row["A_ij"]
                G[t1][t2]["A_ji"] = row["A_ji"]

    # Add GNN-predicted edges not in SPIEC-EASI (high confidence only)
    if gnn_prob_path:
        prob_df = pd.read_csv(gnn_prob_path, sep="\t", index_col=0)
        for t1 in taxa:
            for t2 in taxa:
                if t1 >= t2:
                    continue
                p = prob_df.loc[t1, t2] if (t1 in prob_df.index and t2 in prob_df.columns) else 0
                if p >= gnn_threshold and not G.has_edge(t1, t2):
                    G.add_edge(t1, t2, weight=p, source="gnn",
                               interaction_type="Predicted", sign="unknown")

    return G


# ── Centrality analysis ───────────────────────────────────────────────────────

def compute_centralities(G: nx.Graph) -> pd.DataFrame:
    """
    Compute multiple centrality metrics to identify keystone taxa.

    Keystone taxa = high centrality nodes whose removal destabilizes the community.
    Different metrics capture different types of keystone roles.
    """
    # Use largest connected component for path-based metrics
    lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    centralities = {}

    # Degree: how many partners does this taxon have?
    centralities["degree"] = dict(G.degree())

    # Betweenness: how often on shortest paths (information/flux hub)?
    centralities["betweenness"] = nx.betweenness_centrality(lcc, normalized=True)

    # Closeness: how quickly can this taxon reach others?
    centralities["closeness"] = nx.closeness_centrality(lcc)

    # Eigenvector: connected to other highly connected nodes?
    try:
        centralities["eigenvector"] = nx.eigenvector_centrality_numpy(G)
    except (nx.NetworkXError, np.linalg.LinAlgError):
        # FIX m3: _numpy variant uses NumPy's eigensolver, not power iteration —
        # it raises NetworkXError / LinAlgError, not PowerIterationFailedConvergence.
        centralities["eigenvector"] = {n: 0 for n in G.nodes()}

    # Hub / Authority scores — FIX M5: HITS is designed for directed graphs.
    # On an undirected graph hub == authority == eigenvector centrality, so
    # including both double-counts the same signal in the composite score.
    # Replace authority_score with PageRank, which is meaningful on undirected graphs.
    hubs, _ = nx.hits(G, max_iter=500)
    centralities["hub_score"]  = hubs
    centralities["pagerank"]   = nx.pagerank(G, weight="weight")

    # Weighted degree (strength)
    centralities["strength"] = {n: sum(d["weight"] for _, _, d in G.edges(n, data=True))
                                 for n in G.nodes()}

    df = pd.DataFrame(centralities).fillna(0)
    df.index.name = "taxon"

    # I3: betweenness and closeness are computed on the LCC only. Nodes outside
    # the LCC receive 0 for both metrics via fillna(0), which is semantically
    # correct (zero shortest-path involvement) but deflates their composite
    # keystone score. Flag them explicitly so users aren't misled.
    lcc_nodes = set(max(nx.connected_components(G), key=len))
    df["in_lcc"] = df.index.map(lambda n: n in lcc_nodes)
    n_isolated = (~df["in_lcc"]).sum()
    if n_isolated:
        print(f"  Note: {n_isolated} taxa are outside the largest connected component. "
              "Their betweenness and closeness are 0 by definition, not by ecology. "
              "Interpret their keystone scores with caution.")

    # Composite keystone score (normalized sum of continuous centrality metrics)
    from sklearn.preprocessing import MinMaxScaler
    score_cols = ["degree", "betweenness", "eigenvector", "closeness",
                  "hub_score", "pagerank", "strength"]
    scaler = MinMaxScaler()
    normed = pd.DataFrame(scaler.fit_transform(df[score_cols]),
                          index=df.index, columns=score_cols)
    df["keystone_score"] = normed.mean(axis=1)
    df = df.sort_values("keystone_score", ascending=False)

    return df


# ── Community detection (ecological guilds) ───────────────────────────────────

def detect_guilds(G: nx.Graph, method: str = "leiden") -> dict:
    """
    Detect ecological guilds (functional clusters) in the network.

    Methods:
        leiden    — Leiden algorithm (best quality, requires leidenalg)
        louvain   — Louvain modularity optimization
        infomap   — Information-theoretic flow compression
        spectral  — Spectral clustering (fixed k)
    """
    if method == "leiden":
        try:
            import leidenalg
            import igraph as ig
            ig_graph = ig.Graph.from_networkx(G)
            partition = leidenalg.find_partition(
                ig_graph, leidenalg.ModularityVertexPartition
            )
            nodes = list(G.nodes())
            return {nodes[i]: part for part, members in enumerate(partition)
                    for i in members}
        except ImportError:
            print("leidenalg not installed, falling back to Louvain")
            method = "louvain"

    if method == "louvain":
        from networkx.algorithms.community import louvain_communities
        communities = louvain_communities(G, weight="weight", seed=42)
        return {node: i for i, comm in enumerate(communities) for node in comm}

    if method == "infomap":
        try:
            from infomap import Infomap
            im = Infomap()
            for u, v, d in G.edges(data=True):
                im.add_link(list(G.nodes()).index(u),
                            list(G.nodes()).index(v),
                            d.get("weight", 1))
            im.run()
            nodes = list(G.nodes())
            return {nodes[n.node_id]: n.module_id for n in im.nodes}
        except ImportError:
            print("infomap not installed, falling back to Louvain")
            return detect_guilds(G, method="louvain")

    raise ValueError(f"Unknown method: {method}")


# ── Robustness simulation ──────────────────────────────────────────────────────

def robustness_simulation(G: nx.Graph,
                           strategy: str = "betweenness",
                           n_trials: int = 10) -> pd.DataFrame:
    """
    Simulate community collapse by sequentially removing taxa.
    Tracks: largest connected component size, number of edges.

    Strategies:
        betweenness — remove most central node first (targeted attack)
        degree      — remove highest-degree node first
        random      — random removal (natural extinction baseline)
    """
    results = []
    nodes = list(G.nodes())

    for trial in range(n_trials):
        # FIX m4: betweenness and degree removal orders are deterministic —
        # every trial is identical, wasting (n_trials-1)× compute.
        # Only random removal needs multiple trials for averaging.
        if strategy != "random" and trial > 0:
            break
        H = G.copy()
        removal_order = []

        for step in range(len(nodes)):
            # Compute removal order
            if strategy == "betweenness":
                bc = nx.betweenness_centrality(H)
                target = max(bc, key=bc.get)
            elif strategy == "degree":
                target = max(H.degree(), key=lambda x: x[1])[0]
            elif strategy == "random":
                target = np.random.choice(list(H.nodes()))
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            removal_order.append(target)
            H.remove_node(target)

            if len(H) == 0:
                lcc_size = 0
            else:
                lcc_size = len(max(nx.connected_components(H), key=len))

            results.append({
                "trial": trial,
                "strategy": strategy,
                "step": step + 1,
                "fraction_removed": (step + 1) / len(nodes),
                "lcc_size": lcc_size,
                "lcc_fraction": lcc_size / len(nodes),
                "n_edges": H.number_of_edges(),
                "removed_node": target,
            })

    return pd.DataFrame(results)


# ── Topology tests ─────────────────────────────────────────────────────────────

def test_small_world(G: nx.Graph, n_random: int = 100) -> dict:
    """
    Test whether the network is small-world (Watts-Strogatz property).
    Small-world: high clustering + short path length relative to random.
    sigma > 1 indicates small-world property.
    """
    lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    n, k = lcc.number_of_nodes(), lcc.number_of_edges()

    C_real = nx.average_clustering(lcc, weight="weight")
    L_real = nx.average_shortest_path_length(lcc)

    # Generate Erdos-Renyi random graphs for comparison
    C_rand_list, L_rand_list = [], []
    for _ in range(n_random):
        rg = nx.erdos_renyi_graph(n, p=(2 * k) / (n * (n - 1)))
        if nx.is_connected(rg):
            C_rand_list.append(nx.average_clustering(rg))
            L_rand_list.append(nx.average_shortest_path_length(rg))

    C_rand = np.mean(C_rand_list) if C_rand_list else np.nan
    L_rand = np.mean(L_rand_list) if L_rand_list else np.nan

    gamma = C_real / C_rand if C_rand > 0 else np.nan   # Clustering ratio
    lam   = L_real / L_rand if L_rand > 0 else np.nan   # Path length ratio
    sigma = gamma / lam if lam > 0 else np.nan           # Small-world coefficient

    return {
        "C_real": C_real, "C_random": C_rand, "gamma": gamma,
        "L_real": L_real, "L_random": L_rand, "lambda": lam,
        "sigma": sigma,
        "is_small_world": sigma > 1 if not np.isnan(sigma) else None,
    }


def test_scale_free(G: nx.Graph) -> dict:
    """
    Test whether degree distribution follows a power law (scale-free network).
    Scale-free: hubs dominate, P(k) ~ k^(-gamma)
    """
    from collections import Counter
    degrees = [d for _, d in G.degree() if d > 0]

    deg_counts = Counter(degrees)
    k_vals  = np.array(sorted(deg_counts.keys()), dtype=float)
    counts  = np.array([deg_counts[k] for k in k_vals], dtype=float)

    # FIX L3: stats.linregress raises ValueError when given fewer than 2 points.
    # This happens on regular graphs (every node same degree) or tiny networks.
    # Return a safe non-scale-free result rather than crashing.
    if len(k_vals) < 2:
        print(f"  test_scale_free: only {len(k_vals)} unique degree value(s) — "
              "cannot fit power law. Network is likely regular or near-regular.")
        return {
            "power_law_exponent": float("nan"),
            "r_squared": 0.0,
            "p_value": 1.0,
            "is_scale_free": False,
        }

    slope, intercept, r, p, se = stats.linregress(np.log(k_vals), np.log(counts))
    return {
        "power_law_exponent": -slope,
        "r_squared": r**2,
        "p_value": p,
        "is_scale_free": r**2 > 0.8 and p < 0.05,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Network topology analysis")
    parser.add_argument("--adjacency",    required=True)
    parser.add_argument("--interactions", default=None)
    parser.add_argument("--gnn-probs",    default=None)
    parser.add_argument("--outdir",       default="../../results/analysis")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    G = build_network(args.adjacency, args.interactions, args.gnn_probs)
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Centrality
    print("\nComputing centralities ...")
    cent_df = compute_centralities(G)
    cent_df.to_csv(outdir / "centrality_metrics.tsv", sep="\t")
    print("Top 10 keystone taxa:")
    print(cent_df[["degree", "betweenness", "eigenvector", "keystone_score"]].head(10))

    # Guild detection
    print("\nDetecting ecological guilds ...")
    guilds = detect_guilds(G, method="louvain")
    guild_df = pd.DataFrame.from_dict(guilds, orient="index", columns=["guild"])
    guild_df.index.name = "taxon"
    guild_df.to_csv(outdir / "ecological_guilds.tsv", sep="\t")
    n_guilds = guild_df["guild"].nunique()
    print(f"Found {n_guilds} guilds")
    print(guild_df["guild"].value_counts().head(10))

    # Robustness
    print("\nRunning robustness simulations ...")
    rob_results = []
    for strat in ["betweenness", "degree", "random"]:
        df = robustness_simulation(G, strategy=strat, n_trials=5)
        rob_results.append(df)
    rob_df = pd.concat(rob_results)
    rob_df.to_csv(outdir / "robustness_simulation.tsv", sep="\t", index=False)

    # Topology tests
    print("\nTopology tests ...")
    sw = test_small_world(G, n_random=50)
    sf = test_scale_free(G)
    # FIX m3: is_small_world is None (not False) when sigma is NaN.
    # None is falsy, so the old ternary printed "NO" — misleading for an
    # inconclusive result.
    sw_label = ("YES" if sw["is_small_world"] is True
                else ("INCONCLUSIVE" if sw["is_small_world"] is None else "NO"))
    print(f"Small-world sigma: {sw['sigma']:.3f} → {sw_label}")
    print(f"Scale-free R²: {sf['r_squared']:.3f}, gamma: {sf['power_law_exponent']:.3f}")

    topo_summary = {**sw, **sf}
    pd.Series(topo_summary).to_csv(outdir / "topology_summary.tsv", sep="\t")

    print(f"\nAll outputs saved → {outdir}")


if __name__ == "__main__":
    main()
