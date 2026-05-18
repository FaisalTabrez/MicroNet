#!/usr/bin/env python3
"""
dashboard.py — Interactive Streamlit dashboard for microbial ecological network visualization.

Run: streamlit run dashboard.py

Features:
    - Force-directed network layout with interaction type coloring
    - Keystone taxa rankings
    - Ecological guild visualization
    - Robustness curves
    - Per-taxon interaction profile drilldown
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MicroNet Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# FIX m5: hardcoded "../../results" only works when Streamlit is launched from
# pipeline/05_viz/. Resolve relative to the script file so it works from any CWD.
RESULTS = Path(__file__).resolve().parent.parent.parent / "results"

# ── Color scheme ──────────────────────────────────────────────────────────────
INTERACTION_COLORS = {
    "Mutualism":    "#2ECC71",   # Green
    "Competition":  "#E74C3C",   # Red
    "Parasitism":   "#E67E22",   # Orange
    "Commensalism": "#3498DB",   # Blue
    "Amensalism":   "#9B59B6",   # Purple
    "Predicted":    "#95A5A6",   # Gray
    "Unknown":      "#BDC3C7",   # Light gray
}

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_all_results():
    data = {}

    adj_path = RESULTS / "inference" / "spieceasi_adj_weighted.tsv"
    if adj_path.exists():
        data["adj"] = pd.read_csv(adj_path, sep="\t", index_col=0)

    edge_path = RESULTS / "inference" / "spieceasi_edgelist.tsv"
    if edge_path.exists():
        data["edges"] = pd.read_csv(edge_path, sep="\t")

    inter_path = RESULTS / "inference" / "classified_interactions.tsv"
    if inter_path.exists():
        data["interactions"] = pd.read_csv(inter_path, sep="\t")

    cent_path = RESULTS / "analysis" / "centrality_metrics.tsv"
    if cent_path.exists():
        data["centrality"] = pd.read_csv(cent_path, sep="\t", index_col=0)

    guild_path = RESULTS / "analysis" / "ecological_guilds.tsv"
    if guild_path.exists():
        data["guilds"] = pd.read_csv(guild_path, sep="\t", index_col=0)

    rob_path = RESULTS / "analysis" / "robustness_simulation.tsv"
    if rob_path.exists():
        data["robustness"] = pd.read_csv(rob_path, sep="\t")

    topo_path = RESULTS / "analysis" / "topology_summary.tsv"
    if topo_path.exists():
        # FIX M9: using names= with index_col=0 can leave the column unnamed in
        # some pandas versions. Assign the column explicitly after loading.
        topo = pd.read_csv(topo_path, sep="\t", header=None, index_col=0)
        topo.columns = ["value"]
        data["topology"] = topo

    gnn_path = RESULTS / "gnn" / "predicted_edge_probabilities.tsv"
    if gnn_path.exists():
        data["gnn_probs"] = pd.read_csv(gnn_path, sep="\t", index_col=0)

    return data


def build_nx_graph(data: dict, min_weight: float = 0.0,
                   include_gnn: bool = False, gnn_threshold: float = 0.7) -> nx.Graph:
    # FIX m2: build_nx_graph previously ignored GNN predictions entirely — the
    # sidebar checkbox captured the value but never passed it here, so GNN edges
    # were never added regardless of the checkbox state.
    if "adj" not in data:
        return nx.Graph()

    adj = data["adj"]
    G = nx.Graph()
    G.add_nodes_from(adj.index)

    for i, t1 in enumerate(adj.index):
        for j, t2 in enumerate(adj.columns[i+1:], i+1):
            w = adj.iloc[i, j]
            if abs(w) > min_weight:
                itype = "Unknown"
                if "interactions" in data:
                    match = data["interactions"][
                        ((data["interactions"]["taxon_i_name"] == t1) &
                         (data["interactions"]["taxon_j_name"] == t2)) |
                        ((data["interactions"]["taxon_i_name"] == t2) &
                         (data["interactions"]["taxon_j_name"] == t1))
                    ]
                    if len(match):
                        itype = match.iloc[0]["interaction_type"]
                g1 = int(data["guilds"].loc[t1, "guild"]) if "guilds" in data and t1 in data["guilds"].index else -1
                g2 = int(data["guilds"].loc[t2, "guild"]) if "guilds" in data and t2 in data["guilds"].index else -1
                G.add_edge(t1, t2, weight=abs(w), sign="+" if w > 0 else "-",
                           interaction_type=itype, guild_i=g1, guild_j=g2,
                           is_inter_guild=(g1 != g2))

    # GNN-predicted edges: only added when checkbox is on and file exists
    if include_gnn and "gnn_probs" in data:
        prob_df = data["gnn_probs"]
        for t1 in adj.index:
            for t2 in adj.index:
                if t1 >= t2:
                    continue
                try:
                    p = prob_df.loc[t1, t2]
                except KeyError:
                    continue
                if p >= gnn_threshold and not G.has_edge(t1, t2):
                    G.add_edge(t1, t2, weight=float(p), source="gnn",
                               interaction_type="Predicted", sign="unknown",
                               guild_i=-1, guild_j=-1, is_inter_guild=False)

    return G


def plotly_network(G: nx.Graph, centrality: pd.DataFrame = None,
                   color_by: str = "interaction_type",
                   layout_algo: str = "spring") -> go.Figure:
    """Build interactive Plotly network figure."""
    if G.number_of_nodes() == 0:
        return go.Figure()

    if layout_algo == "spring":
        pos = nx.spring_layout(G, weight="weight", seed=42, k=2/np.sqrt(G.number_of_nodes()))
    elif layout_algo == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G, weight="weight")
    elif layout_algo == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Edges
    edge_traces = []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        itype = d.get("interaction_type", "Unknown")
        color = INTERACTION_COLORS.get(itype, "#BDC3C7")
        width = max(0.5, d.get("weight", 0.1) * 3)
        dash  = "dot" if d.get("sign", "+") == "-" else "solid"
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=width, color=color, dash=dash),
            hoverinfo="none",
            showlegend=False,
        ))

    # Nodes
    node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        deg = G.degree(node)
        size = 8 + deg * 1.5
        if centrality is not None and node in centrality.index:
            ks = centrality.loc[node, "keystone_score"]
            size = 8 + ks * 40
        node_size.append(min(size, 40))
        node_color.append(deg)
        node_text.append(
            f"<b>{node}</b><br>"
            f"Degree: {deg}<br>"
            + (f"Keystone score: {centrality.loc[node, 'keystone_score']:.3f}"
               if centrality is not None and node in centrality.index else "")
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=node_size, color=node_color,
                    colorscale="Viridis", showscale=True,
                    colorbar=dict(title="Degree"),
                    line=dict(width=1, color="white")),
        text=[n.split("|")[-1][:20] for n in G.nodes()],
        textposition="top center", textfont=dict(size=8),
        hovertext=node_text, hoverinfo="text",
        showlegend=False,
    )

    # Legend traces (dummy)
    legend_traces = [
        go.Scatter(x=[None], y=[None], mode="lines",
                   line=dict(color=c, width=3),
                   name=itype, showlegend=True)
        for itype, c in INTERACTION_COLORS.items()
        if any(d.get("interaction_type") == itype for _, _, d in G.edges(data=True))
    ]

    fig = go.Figure(data=edge_traces + legend_traces + [node_trace])
    fig.update_layout(
        title=f"Microbial Ecological Network  ({G.number_of_nodes()} taxa, {G.number_of_edges()} interactions)",
        showlegend=True,
        legend=dict(title="Interaction Type", x=1.01),
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=650,
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="white"),
    )
    return fig


# ── Main UI ───────────────────────────────────────────────────────────────────

def main():
    st.title("🦠 MicroNet — Microbial Ecological Network Dashboard")
    st.markdown("*Inferring cooperation, competition, and symbiosis from metagenomics*")

    data = load_all_results()

    if not data:
        st.warning("No results found. Run the pipeline first.")
        st.code("snakemake --cores 8\npython pipeline/02_inference/glv_inference.py ...\npython pipeline/03_gnn/train_vgae.py ...")
        return

    # ── Sidebar controls ──────────────────────────────────────────────────────
    st.sidebar.header("Network Controls")
    min_weight = st.sidebar.slider("Min edge weight", 0.0, 1.0, 0.05, 0.01)
    layout_algo = st.sidebar.selectbox("Layout", ["spring", "kamada_kawai", "circular"])
    max_nodes = st.sidebar.slider("Max nodes (top by degree)", 10, 200, 80)
    show_gnn = st.sidebar.checkbox("Include GNN-predicted edges", value=False)
    gnn_threshold = st.sidebar.slider("GNN confidence threshold", 0.5, 1.0, 0.7, 0.05,
                                       disabled=not show_gnn)

    # FIX m2: pass show_gnn so the checkbox actually controls GNN edge inclusion
    G = build_nx_graph(data, min_weight=min_weight,
                       include_gnn=show_gnn, gnn_threshold=gnn_threshold)

    # Restrict to top N nodes by degree
    if G.number_of_nodes() > max_nodes:
        top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_names = [n for n, _ in top_nodes]
        G = G.subgraph(top_names).copy()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌐 Network", "🔑 Keystone Taxa", "🏘️ Guilds",
        "💪 Robustness", "📊 Topology"
    ])

    with tab1:
        cent = data.get("centrality")
        fig = plotly_network(G, centrality=cent, layout_algo=layout_algo)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Taxa (nodes)", G.number_of_nodes())
        col2.metric("Interactions (edges)", G.number_of_edges())
        col3.metric("Network density", f"{nx.density(G):.4f}")
        col4.metric("Avg. clustering", f"{nx.average_clustering(G):.4f}")

        if "interactions" in data:
            itype_counts = data["interactions"]["interaction_type"].value_counts()
            fig_pie = px.pie(values=itype_counts.values,
                             names=itype_counts.index,
                             color=itype_counts.index,
                             color_discrete_map=INTERACTION_COLORS,
                             title="Interaction Type Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)

    with tab2:
        if "centrality" in data:
            cent_df = data["centrality"]
            st.subheader("🔑 Keystone Taxa (composite centrality score)")

            fig_ks = px.bar(
                cent_df.head(20).reset_index(),
                x="taxon", y="keystone_score",
                color="betweenness", color_continuous_scale="Viridis",
                title="Top 20 Keystone Taxa",
            )
            fig_ks.update_xaxes(tickangle=45)
            st.plotly_chart(fig_ks, use_container_width=True)

            st.dataframe(
                cent_df.head(30)[["degree", "betweenness", "eigenvector",
                                   "closeness", "pagerank", "keystone_score"]].round(4),
                use_container_width=True
            )
        else:
            st.info("Run topology.py to generate centrality metrics.")

    with tab3:
        if "guilds" in data:
            guild_df = data["guilds"]
            guild_counts = guild_df["guild"].value_counts().reset_index()
            guild_counts.columns = ["guild", "n_taxa"]

            fig_g = px.bar(guild_counts.head(15), x="guild", y="n_taxa",
                           title="Ecological Guild Sizes",
                           labels={"guild": "Guild ID", "n_taxa": "Number of Taxa"},
                           color="n_taxa", color_continuous_scale="Blues")
            st.plotly_chart(fig_g, use_container_width=True)

            selected_guild = st.selectbox("Explore guild", sorted(guild_df["guild"].unique()))
            members = guild_df[guild_df["guild"] == selected_guild].index.tolist()
            st.write(f"**Guild {selected_guild}** — {len(members)} taxa")
            st.write(", ".join(members[:50]))
        else:
            st.info("Run topology.py to detect ecological guilds.")

    with tab4:
        if "robustness" in data:
            rob_df = data["robustness"]
            fig_rob = px.line(
                rob_df.groupby(["strategy", "fraction_removed"])["lcc_fraction"].mean().reset_index(),
                x="fraction_removed", y="lcc_fraction", color="strategy",
                title="Community Robustness — LCC size as fraction of original network",
                labels={"fraction_removed": "Fraction of taxa removed",
                        "lcc_fraction": "LCC / original network size"},
                color_discrete_map={
                    "betweenness": "#E74C3C",
                    "degree": "#E67E22",
                    "random": "#3498DB",
                }
            )
            fig_rob.add_hline(y=0.5, line_dash="dash", line_color="gray",
                              annotation_text="50% collapse threshold")
            st.plotly_chart(fig_rob, use_container_width=True)
            st.info("Red (betweenness): targeted attack on keystone taxa. Blue (random): baseline extinction.")
        else:
            st.info("Run topology.py to generate robustness simulations.")

    with tab5:
        if "topology" in data:
            topo = data["topology"]
            st.subheader("Network Topology Summary")

            def safe_float(key: str, default: float = float("nan")) -> float:
                """Guard against NaN/None serialised as empty strings in the TSV."""
                try:
                    return float(topo.loc[key, "value"])
                except (ValueError, KeyError):
                    return default

            col1, col2 = st.columns(2)
            with col1:
                sigma = safe_float("sigma")
                sw_label = ("✅ Yes" if sigma > 1 else ("❓ Inconclusive" if np.isnan(sigma) else "❌ No"))
                st.metric("Small-world σ", f"{sigma:.3f}" if not np.isnan(sigma) else "N/A",
                          delta=sw_label, delta_color="off")
                st.metric("Clustering (real)",   f"{safe_float('C_real'):.4f}")
                st.metric("Clustering (random)", f"{safe_float('C_random'):.4f}")
            with col2:
                r2 = safe_float("r_squared")
                sf_label = "✅ Yes" if r2 > 0.8 else "❌ No"
                st.metric("Scale-free R²", f"{r2:.3f}", delta=sf_label, delta_color="off")
                st.metric("Power-law exponent γ", f"{safe_float('power_law_exponent'):.3f}")
                st.metric("Path length (real)",    f"{safe_float('L_real'):.3f}")
        else:
            st.info("Run topology.py to compute topology statistics.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**MicroNet Pipeline**")
    st.sidebar.markdown("[GitHub](https://github.com) | [Docs](https://docs)")


if __name__ == "__main__":
    main()
