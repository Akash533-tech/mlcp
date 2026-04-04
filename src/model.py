"""
src/model.py
============
GNN recommendation models:

  1. LightGCN   – linear graph convolution with no non-linearity;
                  state-of-the-art for collaborative filtering.
  2. GraphSAGE  – inductive convolution with mean aggregation + ReLU;
                  better for cold-start / feature-rich settings.

Both models output (user_embs, movie_embs) and support dot-product scoring.
BPR and MSE losses are provided as standalone functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, LGConv


# ─────────────────────────────────────────────────────────────────────────────
class LightGCN(nn.Module):
    """
    He et al. (2020) – "LightGCN: Simplifying and Powering Graph
    Convolution Network for Recommendation."

    Architecture
    ------------
    - Learnable embeddings for users and movies (no input features).
    - K graph convolution layers (each is simply a neighbourhood average).
    - Final embedding = mean of embeddings across all layers (including layer 0).
    - Score = dot product of user_emb and movie_emb.
    """

    def __init__(self,
                 n_users:  int,
                 n_movies: int,
                 emb_dim:  int  = 64,
                 n_layers: int  = 3,
                 dropout:  float = 0.1):
        super().__init__()
        self.n_users  = n_users
        self.n_movies = n_movies
        self.emb_dim  = emb_dim
        self.n_layers = n_layers
        self.dropout  = dropout

        self.user_emb  = nn.Embedding(n_users,  emb_dim)
        self.movie_emb = nn.Embedding(n_movies, emb_dim)

        self.convs = nn.ModuleList([LGConv() for _ in range(n_layers)])

        nn.init.normal_(self.user_emb.weight,  std=0.1)
        nn.init.normal_(self.movie_emb.weight, std=0.1)

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, edge_index: torch.Tensor):
        """
        Parameters
        ----------
        edge_index : LongTensor [2, E]
            Bipartite graph edges in the combined user+movie space.
            User nodes are indexed [0, n_users-1],
            Movie nodes are indexed [n_users, n_users+n_movies-1].

        Returns
        -------
        user_embs, movie_embs : FloatTensor [n_users, D], [n_movies, D]
        """
        x = torch.cat([self.user_emb.weight, self.movie_emb.weight], dim=0)

        # Edge dropout during training
        if self.training and self.dropout > 0:
            mask = torch.rand(edge_index.size(1), device=edge_index.device)
            edge_index = edge_index[:, mask > self.dropout]

        # Propagate
        all_embs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            all_embs.append(x)

        # Final = mean across layers
        final = torch.stack(all_embs, dim=1).mean(dim=1)
        user_embs  = final[:self.n_users]
        movie_embs = final[self.n_users:]
        return user_embs, movie_embs

    # ── scoring ───────────────────────────────────────────────────────────────
    def score_users_movies(self,
                           edge_index: torch.Tensor,
                           user_ids:   torch.Tensor,
                           movie_ids:  torch.Tensor) -> torch.Tensor:
        user_embs, movie_embs = self.forward(edge_index)
        u = user_embs[user_ids]
        m = movie_embs[movie_ids]
        return (u * m).sum(dim=-1)

    def recommend(self,
                  edge_index: torch.Tensor,
                  user_id:    int,
                  top_k:      int = 10,
                  exclude:    set = None) -> torch.Tensor:
        """Return indices of top-K recommended movies for a user."""
        self.eval()
        with torch.no_grad():
            user_embs, movie_embs = self.forward(edge_index)
            scores = movie_embs @ user_embs[user_id]  # [n_movies]
            if exclude:
                scores[list(exclude)] = float("-inf")
            return scores.topk(top_k).indices


# ─────────────────────────────────────────────────────────────────────────────
class GraphSAGERecommender(nn.Module):
    """
    GraphSAGE-based recommender with optional input node features.
    Useful for cold-start (feature-rich) settings.
    """

    def __init__(self,
                 n_users:      int,
                 n_movies:     int,
                 in_channels:  int  = 64,
                 hidden_dim:   int  = 128,
                 out_dim:      int  = 64,
                 n_layers:     int  = 2,
                 dropout:      float = 0.2):
        super().__init__()
        self.n_users  = n_users
        self.n_movies = n_movies
        self.dropout  = dropout

        self.user_emb  = nn.Embedding(n_users,  in_channels)
        self.movie_emb = nn.Embedding(n_movies, in_channels)

        dims = [in_channels] + [hidden_dim] * (n_layers - 1) + [out_dim]
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(SAGEConv(dims[i], dims[i + 1]))

        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.movie_emb.weight)

    def forward(self, edge_index: torch.Tensor):
        x = torch.cat([self.user_emb.weight, self.movie_emb.weight], dim=0)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        user_embs  = x[:self.n_users]
        movie_embs = x[self.n_users:]
        return user_embs, movie_embs

    def recommend(self,
                  edge_index: torch.Tensor,
                  user_id:    int,
                  top_k:      int = 10,
                  exclude:    set = None) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            user_embs, movie_embs = self.forward(edge_index)
            scores = movie_embs @ user_embs[user_id]
            if exclude:
                scores[list(exclude)] = float("-inf")
            return scores.topk(top_k).indices


# ─────────────────────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────────────────────

def bpr_loss(user_embs:  torch.Tensor,
             pos_embs:   torch.Tensor,
             neg_embs:   torch.Tensor,
             l2_lambda:  float = 1e-4) -> torch.Tensor:
    """
    Bayesian Personalised Ranking loss.

    Parameters
    ----------
    user_embs : [B, D]
    pos_embs  : [B, D]   – positively rated movie embeddings
    neg_embs  : [B, D]   – randomly sampled negative movie embeddings
    """
    pos_scores = (user_embs * pos_embs).sum(dim=-1)   # [B]
    neg_scores = (user_embs * neg_embs).sum(dim=-1)   # [B]
    loss = -F.logsigmoid(pos_scores - neg_scores).mean()
    # L2 regularisation
    l2 = (user_embs.norm(2).pow(2) +
          pos_embs.norm(2).pow(2) +
          neg_embs.norm(2).pow(2)) / (2 * len(user_embs))
    return loss + l2_lambda * l2


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standard MSE for explicit rating prediction."""
    return F.mse_loss(pred, target)
