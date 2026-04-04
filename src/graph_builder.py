"""
src/graph_builder.py
====================
Constructs a heterogeneous interaction graph from preprocessed MovieLens data.

Nodes  : User,  Movie,  Genre
Edges  : User  --[rated]--> Movie       (weighted by rating, decayed by time)
         Movie --[belongs_to]--> Genre
         User  --[similar_to]--> User   (K-NN on rating patterns, k=10)
         Movie --[co_watched]-->  Movie  (co-occurrence threshold ≥ 5 shared users)

Returns both a NetworkX graph (for visualisation / Node2Vec) and
a PyTorch Geometric Data object (for GNN training).
"""

import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import HeteroData
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from collections import defaultdict
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
def time_decay(timestamps: np.ndarray, half_life_days: float = 365.0) -> np.ndarray:
    """Exponential time-decay: recent interactions get weight ≈ 1."""
    max_ts = timestamps.max()
    delta  = (max_ts - timestamps) / (half_life_days * 86400)
    return np.exp(-delta)


# ─────────────────────────────────────────────────────────────────────────────
def build_networkx_graph(data: dict) -> nx.Graph:
    """
    Build a NetworkX graph for visualisation and Node2Vec pre-training.

    Node naming convention:
        "u_{user_idx}"   – user nodes
        "m_{movie_idx}"  – movie nodes
        "g_{genre}"      – genre nodes
    """
    ratings   = data["ratings"]
    movies    = data["movies"]
    genre_list = data["genre_list"]
    movie2idx  = data["movie2idx"]

    # Build a *valid movie set* (movies that appear in ratings)
    rated_movie_ids = set(ratings["movieId"].unique())

    G = nx.Graph()

    # ── User nodes ────────────────────────────────────────────────────────────
    for uid, idx in data["user2idx"].items():
        G.add_node(f"u_{idx}", node_type="user", original_id=uid)

    # ── Movie nodes ───────────────────────────────────────────────────────────
    movie_meta = movies.set_index("movieId")
    for mid, idx in movie2idx.items():
        title = movie_meta.loc[mid, "title"] if mid in movie_meta.index else f"Movie {mid}"
        year  = movie_meta.loc[mid, "year"]  if mid in movie_meta.index else 0
        G.add_node(f"m_{idx}", node_type="movie", original_id=mid,
                   title=title, year=year)

    # ── Genre nodes ───────────────────────────────────────────────────────────
    for g in genre_list:
        G.add_node(f"g_{g}", node_type="genre", genre=g)

    # ── User–Movie edges (rated) ──────────────────────────────────────────────
    decay_weights = time_decay(ratings["timestamp"].values)
    for (_, row), w in zip(ratings.iterrows(), decay_weights):
        G.add_edge(f"u_{int(row['user_idx'])}",
                   f"m_{int(row['movie_idx'])}",
                   edge_type="rated",
                   rating=row["rating"],
                   weight=float(row["rating"]) * w)

    # ── Movie–Genre edges ─────────────────────────────────────────────────────
    for _, row in movies.iterrows():
        if row["movieId"] not in movie2idx:
            continue
        mid_idx = movie2idx[row["movieId"]]
        for g in row["genres"]:
            if G.has_node(f"g_{g}"):
                G.add_edge(f"m_{mid_idx}", f"g_{g}",
                           edge_type="belongs_to", weight=1.0)

    # ── Co-watched Movie–Movie edges ──────────────────────────────────────────
    print("  Computing co-watched edges …")
    movie_users = defaultdict(set)
    for _, row in ratings.iterrows():
        movie_users[int(row["movie_idx"])].add(int(row["user_idx"]))

    movie_ids = list(movie_users.keys())
    THRESHOLD = 5
    for i in range(len(movie_ids)):
        for j in range(i + 1, len(movie_ids)):
            shared = len(movie_users[movie_ids[i]] & movie_users[movie_ids[j]])
            if shared >= THRESHOLD:
                G.add_edge(f"m_{movie_ids[i]}", f"m_{movie_ids[j]}",
                           edge_type="co_watched", weight=float(shared))

    # ── User–User similarity (K-NN, k=10) ────────────────────────────────────
    print("  Computing user-user similarity edges …")
    n_users  = data["n_users"]
    n_movies = data["n_movies"]

    row_idx = ratings["user_idx"].values.astype(int)
    col_idx = ratings["movie_idx"].values.astype(int)
    vals    = ratings["rating"].values.astype(float)

    user_movie_mat = csr_matrix((vals, (row_idx, col_idx)),
                                shape=(n_users, n_movies))

    # Sample top-k similar users via cosine similarity in batches
    K = 10
    BATCH = 100
    for start in range(0, n_users, BATCH):
        end  = min(start + BATCH, n_users)
        sim  = cosine_similarity(user_movie_mat[start:end], user_movie_mat)
        for local_i, full_i in enumerate(range(start, end)):
            sims_row = sim[local_i]
            sims_row[full_i] = -1  # exclude self
            top_k = np.argsort(sims_row)[-K:]
            for j in top_k:
                if sims_row[j] > 0:
                    G.add_edge(f"u_{full_i}", f"u_{j}",
                               edge_type="similar_to",
                               weight=float(sims_row[j]))

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ─────────────────────────────────────────────────────────────────────────────
def build_pyg_data(data: dict) -> HeteroData:
    """
    Build a PyTorch Geometric HeteroData object for GNN training.

    Only User and Movie nodes + their relationships are included here
    (genre features are encoded as multi-hot vectors on Movies).
    """
    ratings   = data["ratings"]
    movies    = data["movies"]
    genre_list = data["genre_list"]
    movie2idx  = data["movie2idx"]
    n_users   = data["n_users"]
    n_movies  = data["n_movies"]
    n_genres  = len(genre_list)
    genre2idx = {g: i for i, g in enumerate(genre_list)}

    pyg = HeteroData()

    # ── Node features ─────────────────────────────────────────────────────────
    # Users: one-hot placeholder (embedding will be learned)
    pyg["user"].x    = torch.arange(n_users).unsqueeze(1).float()   # [n_users, 1]
    pyg["user"].num_nodes = n_users

    # Movies: multi-hot genre vector + normalised year
    movie_meta = movies.set_index("movieId")
    year_min, year_max = movies["year"].min(), movies["year"].max()
    year_range = max(year_max - year_min, 1)

    movie_feats = np.zeros((n_movies, n_genres + 1), dtype=np.float32)
    for mid, midx in movie2idx.items():
        if mid in movie_meta.index:
            row = movie_meta.loc[mid]
            for g in row["genres"]:
                if g in genre2idx:
                    movie_feats[midx, genre2idx[g]] = 1.0
            movie_feats[midx, n_genres] = (row["year"] - year_min) / year_range
    pyg["movie"].x    = torch.tensor(movie_feats)
    pyg["movie"].num_nodes = n_movies

    # Genres: one-hot identity
    pyg["genre"].x    = torch.eye(n_genres)
    pyg["genre"].num_nodes = n_genres

    # ── Edges: User → Movie (rated) ───────────────────────────────────────────
    decay_w = time_decay(ratings["timestamp"].values)
    edge_rating = torch.tensor(ratings["rating"].values, dtype=torch.float)
    edge_w      = torch.tensor(
        ratings["rating"].values * decay_w, dtype=torch.float)

    user_idxs  = torch.tensor(ratings["user_idx"].values,  dtype=torch.long)
    movie_idxs = torch.tensor(ratings["movie_idx"].values, dtype=torch.long)

    pyg["user", "rated", "movie"].edge_index = torch.stack([user_idxs, movie_idxs])
    pyg["user", "rated", "movie"].edge_attr  = edge_w.unsqueeze(1)
    pyg["user", "rated", "movie"].rating     = edge_rating

    # Reverse edge for message passing
    pyg["movie", "rated_by", "user"].edge_index = torch.stack([movie_idxs, user_idxs])
    pyg["movie", "rated_by", "user"].edge_attr  = edge_w.unsqueeze(1)

    # ── Edges: Movie → Genre ──────────────────────────────────────────────────
    mg_src, mg_dst = [], []
    for mid, midx in movie2idx.items():
        if mid in movie_meta.index:
            for g in movie_meta.loc[mid, "genres"]:
                if g in genre2idx:
                    mg_src.append(midx)
                    mg_dst.append(genre2idx[g])
    if mg_src:
        pyg["movie", "belongs_to", "genre"].edge_index = torch.tensor(
            [mg_src, mg_dst], dtype=torch.long)
        pyg["genre", "contains", "movie"].edge_index = torch.tensor(
            [mg_dst, mg_src], dtype=torch.long)

    return pyg
