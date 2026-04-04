"""
src/visualize.py
================
Graph and metric visualisation helpers.

Functions
---------
  plot_rating_distribution(ratings)        – histogram of rating values
  plot_genre_distribution(movies)          – bar chart of genre frequency
  plot_sparsity_heatmap(ratings, ...)      – user×movie rating matrix heat-map
  plot_training_history(history)           – loss / metrics vs epoch curves
  visualize_graph_pyvis(G, output_html)    – interactive PyVis HTML graph
  draw_subgraph(G, node, hops, ax)         – NetworkX n-hop neighbourhood
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from collections import Counter


# Colour palette ──────────────────────────────────────────────────────────────
PALETTE = {
    "user":       "#6C63FF",
    "movie":      "#FF6584",
    "genre":      "#43E97B",
    "text":       "#E8E8FF",
    "bg":         "#0F0F1A",
    "grid":       "#1E1E3A",
    "accent":     "#FFBE0B",
}

plt.rcParams.update({
    "figure.facecolor": PALETTE["bg"],
    "axes.facecolor":   PALETTE["grid"],
    "axes.edgecolor":   PALETTE["text"],
    "text.color":       PALETTE["text"],
    "axes.labelcolor":  PALETTE["text"],
    "xtick.color":      PALETTE["text"],
    "ytick.color":      PALETTE["text"],
    "grid.color":       "#2A2A4A",
    "font.family":      "DejaVu Sans",
})


# ─────────────────────────────────────────────────────────────────────────────
def plot_rating_distribution(ratings: pd.DataFrame, ax=None):
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    counts = ratings["rating"].value_counts().sort_index()
    bars = ax.bar(counts.index.astype(str), counts.values,
                  color=PALETTE["accent"], edgecolor="#000", width=0.6, alpha=0.85)
    ax.set_title("Rating Distribution", fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.4)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 200,
                f"{int(h):,}", ha="center", va="bottom", fontsize=8)
    if show:
        plt.tight_layout(); plt.show()
    return ax


# ─────────────────────────────────────────────────────────────────────────────
def plot_genre_distribution(movies: pd.DataFrame, ax=None):
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    genre_counts = Counter(g for genres in movies["genres"] for g in genres)
    labels, vals = zip(*sorted(genre_counts.items(), key=lambda x: -x[1]))
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(labels)))
    ax.barh(labels, vals, color=colors, edgecolor="#000", alpha=0.85)
    ax.set_title("Genre Frequency", fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Movie Count")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.4)
    if show:
        plt.tight_layout(); plt.show()
    return ax


# ─────────────────────────────────────────────────────────────────────────────
def plot_sparsity_heatmap(ratings: pd.DataFrame,
                          n_users_sample: int = 50,
                          n_movies_sample: int = 80,
                          ax=None):
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    sample_users  = sorted(ratings["user_idx"].unique())[:n_users_sample]
    sample_movies = sorted(ratings["movie_idx"].unique())[:n_movies_sample]

    mat = np.zeros((len(sample_users), len(sample_movies)), dtype=float)
    u_map = {u: i for i, u in enumerate(sample_users)}
    m_map = {m: i for i, m in enumerate(sample_movies)}

    sub = ratings[ratings["user_idx"].isin(sample_users) &
                  ratings["movie_idx"].isin(sample_movies)]
    for _, row in sub.iterrows():
        mat[u_map[int(row["user_idx"])], m_map[int(row["movie_idx"])]] = row["rating"]

    im = ax.imshow(mat, cmap="plasma", aspect="auto", interpolation="nearest")
    ax.set_title(f"Rating Matrix (first {n_users_sample} users × {n_movies_sample} movies)",
                 fontsize=13, pad=10)
    ax.set_xlabel("Movie index")
    ax.set_ylabel("User index")
    plt.colorbar(im, ax=ax, label="Rating")
    if show:
        plt.tight_layout(); plt.show()
    return ax


# ─────────────────────────────────────────────────────────────────────────────
def plot_training_history(history: list, ax=None):
    show = ax is None
    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    else:
        axes = ax

    epochs = [h["epoch"] for h in history]
    losses  = [h["train_loss"]     for h in history]
    recalls = [h["recall_at_k"]    for h in history]
    ndcgs   = [h["ndcg_at_k"]      for h in history]

    for ax_, y, label, color in [
        (axes[0], losses,  "BPR Loss",   PALETTE["accent"]),
        (axes[1], recalls, "Recall@10",  PALETTE["user"]),
        (axes[2], ndcgs,   "NDCG@10",   PALETTE["genre"]),
    ]:
        ax_.plot(epochs, y, color=color, linewidth=2, marker="o", markersize=5)
        ax_.set_title(label, fontsize=12, fontweight="bold")
        ax_.set_xlabel("Epoch")
        ax_.grid(alpha=0.4)

    if show:
        plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────────────────────────────────────
def visualize_graph_pyvis(G: nx.Graph,
                          output_html: str = "graph_vis.html",
                          max_nodes: int = 300,
                          max_edges: int = 800):
    """
    Generate an interactive PyVis HTML graph.
    Samples up to max_nodes nodes for performance.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        print("pyvis not installed – skipping interactive visualisation")
        return

    net = Network(height="700px", width="100%",
                  bgcolor=PALETTE["bg"], font_color=PALETTE["text"],
                  notebook=False)
    net.barnes_hut(gravity=-3000, central_gravity=0.3,
                   spring_length=120, damping=0.09)

    sampled_nodes = list(G.nodes)[:max_nodes]
    sub = G.subgraph(sampled_nodes)

    color_map = {"user": PALETTE["user"],
                 "movie": PALETTE["movie"],
                 "genre": PALETTE["genre"]}
    size_map  = {"user": 10, "movie": 8, "genre": 14}

    for n, attrs in sub.nodes(data=True):
        ntype = attrs.get("node_type", "movie")
        label = attrs.get("title", attrs.get("genre", str(n)))[:25]
        net.add_node(str(n), label=label,
                     color=color_map.get(ntype, "#FFFFFF"),
                     size=size_map.get(ntype, 8),
                     title=f"{ntype}: {label}")

    edges_added = 0
    for u, v, edata in sub.edges(data=True):
        if edges_added >= max_edges:
            break
        etype = edata.get("edge_type", "")
        w     = edata.get("weight", 1.0)
        net.add_edge(str(u), str(v),
                     title=f"{etype} ({w:.2f})",
                     width=max(0.5, min(w, 3.0)))
        edges_added += 1

    net.save_graph(output_html)
    print(f"  Interactive graph saved → {output_html}")


# ─────────────────────────────────────────────────────────────────────────────
def draw_subgraph(G: nx.Graph,
                  center_node: str,
                  hops: int = 2,
                  ax=None):
    """
    Draw the n-hop neighbourhood of `center_node` in NetworkX.
    """
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # BFS to collect nodes within `hops` hops
    nodes = {center_node}
    frontier = {center_node}
    for _ in range(hops):
        next_frontier = set()
        for n in frontier:
            next_frontier.update(G.neighbors(n))
        nodes |= next_frontier
        frontier = next_frontier

    sub = G.subgraph(nodes)
    pos = nx.spring_layout(sub, seed=42)

    color_map_attr = {"user": PALETTE["user"],
                      "movie": PALETTE["movie"],
                      "genre": PALETTE["genre"]}
    node_colors = [
        color_map_attr.get(sub.nodes[n].get("node_type", "movie"), "#AAAAAA")
        for n in sub.nodes
    ]
    node_sizes = [300 if n == center_node else 100 for n in sub.nodes]

    nx.draw_networkx(sub, pos=pos, ax=ax,
                     node_color=node_colors,
                     node_size=node_sizes,
                     with_labels=True,
                     labels={n: sub.nodes[n].get("title", n)[:12] for n in sub.nodes},
                     font_size=6,
                     edge_color=PALETTE["text"],
                     alpha=0.8)

    legend_patches = [
        mpatches.Patch(color=c, label=t)
        for t, c in color_map_attr.items()
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=8)
    ax.set_title(f"{hops}-hop neighbourhood of {center_node}", fontsize=12)
    ax.axis("off")

    if show:
        plt.tight_layout(); plt.show()
    return ax
