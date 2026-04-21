import torch
import torch.nn as nn
from torch_geometric.explain import Explainer, GNNExplainer
from pyvis.network import Network
import tempfile
import os
import pandas as pd
import numpy as np

class LightGCNLinkPredictionWrapper(nn.Module):
    """
    Wraps the LightGCN model for PyTorch Geometric's Explainer API.
    The Explainer expects a single forward pass that returns a 1D score per edge,
    so we process `edge_index` to get embeddings, and then dot-product the `edge_label_index`.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, edge_label_index):
        # x is ignored because LightGCN relies strictly on internal nn.Embedding lookup
        user_embs, movie_embs = self.model(edge_index)
        
        # edge_label_index represents the specific link(s) to score: src=user, dst=movie
        users = edge_label_index[0]
        movies = edge_label_index[1] - self.model.n_users
        
        u = user_embs[users]
        m = movie_embs[movies]
        return (u * m).sum(dim=-1)


def explain_recommendation(model, edge_index, n_users, user_idx, movie_idx, top_k_edges=20):
    """
    Explains why a specific `movie_idx` was recommended to `user_idx`.
    """
    wrapper = LightGCNLinkPredictionWrapper(model)
    wrapper.eval()

    # Target link prediction to explain
    target_link = torch.tensor([[user_idx], [movie_idx + n_users]], dtype=torch.long)

    t_edge_index = edge_index.to(target_link.device)
    
    # We configure a node-independent edge explainer leveraging regression on the prediction score
    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=150),
        explanation_type='model',
        node_mask_type=None,
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='edge',
            return_type='raw',
        ),
    )

    # dummy node feature
    dummy_x = torch.empty((n_users + model.n_movies, 1), device=target_link.device)

    explanation = explainer(
        x=dummy_x, 
        edge_index=t_edge_index, 
        edge_label_index=target_link
    )
    
    # Extract the masked values representing component edge importance
    edge_mask = explanation.edge_mask.detach().cpu().numpy()
    
    # Identify unique undirected influential paths
    # Because our input edge_index is bipartite bi-directional, we map (src, dst) to sets
    top_indices = edge_mask.argsort()[::-1]
    
    important_edges = []
    seen_pairs = set()
    
    for idx in top_indices:
        src = int(edge_index[0, idx])
        dst = int(edge_index[1, idx])
        score = float(edge_mask[idx])
        
        pair = frozenset([src, dst])
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            important_edges.append((src, dst, score))
            
        if len(important_edges) >= top_k_edges:
            break
            
    return important_edges


def generate_explanation_html(important_edges, data, user_idx, target_movie_idx):
    """
    Generates an interactive PyVis network HTML string that illustrates
    the most influential edges driving the prediction.
    """
    movies     = data["movies"]
    idx2movie  = {v: k for k, v in data["movie2idx"].items()}
    n_users    = data["n_users"]
    
    net = Network(height="400px", width="100%", bgcolor="#12122a", font_color="#e8e8ff")
    
    def get_node_label(node_idx):
        if node_idx < n_users:
            return f"User {node_idx}", "ellipse", "#6C63FF"
        else:
            midx = node_idx - n_users
            mid = idx2movie.get(midx)
            if mid is not None:
                row = movies[movies["movieId"] == mid]
                if not row.empty:
                    title = row.iloc[0]["title"]
                    year = row.iloc[0]["year"]
                    if len(title) > 23: title = title[:20] + "..."
                    return title + f" ({year})", "box", "#43E97B"
            return f"Movie #{midx}", "box", "#43E97B"

    # Guarantee the target nodes exist with distinctive style (User -> Target Movie)
    t_u_lbl, t_u_shp, _ = get_node_label(user_idx)
    t_m_lbl, t_m_shp, _ = get_node_label(target_movie_idx + n_users)
    
    net.add_node(user_idx, label=t_u_lbl, shape=t_u_shp, color="#FFBE0B", size=30, title="Target User", borderWidth=3)
    net.add_node(target_movie_idx + n_users, label=t_m_lbl, shape=t_m_shp, color="#FF6584", size=30, title="Recommended Movie", borderWidth=3)

    min_score = min(score for _, _, score in important_edges)
    max_score = max(score for _, _, score in important_edges)
    range_score = max(max_score - min_score, 1e-5)

    for src, dst, score in important_edges:
        u, m_node = (src, dst) if src < n_users else (dst, src)
        
        for node in (u, m_node):
            if node not in net.get_nodes():
                lbl, shp, col = get_node_label(node)
                net.add_node(node, label=lbl, shape=shp, color=col)
                
        # Enhance visual thickness relative to others
        norm_score = (score - min_score) / range_score
        thickness = 1 + norm_score * 4
        
        net.add_edge(u, m_node, value=thickness, title=f"Influence Score: {score:.4f}", color="#a8a0ff")
        
    net.toggle_physics(True)
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      }
    }
    """)

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".html", delete=False) as f:
        net.save_graph(f.name)
        f.seek(0)
        html_data = f.read()
    
    try:
        os.remove(f.name)
    except:
        pass
        
    return html_data
