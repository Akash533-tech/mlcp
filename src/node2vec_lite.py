"""
src/node2vec_lite.py
====================
Self-contained, dependency-free Node2Vec implementation using PyTorch.

We implement random walks (biased by the p/q parameters) over a NetworkX graph,
then train a skip-gram model using negative sampling to produce node embeddings.

This replaces the `node2vec` pip package (which requires gensim, incompatible
with Python 3.13).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import random
from tqdm import tqdm
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# Biased Random Walk
# ─────────────────────────────────────────────────────────────────────────────

def _alias_setup(probs):
    """Alias method for O(1) sampling from a discrete distribution."""
    K   = len(probs)
    q   = np.array(probs, dtype=np.float64)
    J   = np.zeros(K, dtype=np.int64)
    smaller, larger = [], []
    for kk, prob in enumerate(q):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)
    while smaller and larger:
        small = smaller.pop()
        large = larger.pop()
        J[small] = large
        q[large] = q[large] - (1.0 - q[small])
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return J, q


def _alias_draw(J, q):
    K  = len(J)
    kk = int(np.floor(np.random.uniform() * K))
    if np.random.uniform() < q[kk]:
        return kk
    return J[kk]


def _get_alias_edge(G, src, dst, p, q):
    unnorm = []
    for nbr in sorted(G.neighbors(dst)):
        if nbr == src:
            unnorm.append(G[dst][nbr].get("weight", 1.0) / p)
        elif G.has_edge(nbr, src):
            unnorm.append(G[dst][nbr].get("weight", 1.0))
        else:
            unnorm.append(G[dst][nbr].get("weight", 1.0) / q)
    total = sum(unnorm)
    probs = [x / total for x in unnorm]
    return _alias_setup(probs)


def biased_random_walk(G: nx.Graph,
                       start_node,
                       walk_length: int,
                       p: float,
                       q: float,
                       alias_nodes: dict,
                       alias_edges: dict) -> list:
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        neighbors = sorted(G.neighbors(cur))
        if not neighbors:
            break
        if len(walk) == 1:
            J, prob = alias_nodes[cur]
            idx = _alias_draw(J, prob)
            walk.append(neighbors[idx])
        else:
            prev = walk[-2]
            J, prob = alias_edges[(prev, cur)]
            idx     = _alias_draw(J, prob)
            walk.append(neighbors[idx])
    return walk


# ─────────────────────────────────────────────────────────────────────────────
# Skip-gram with Negative Sampling
# ─────────────────────────────────────────────────────────────────────────────

class SkipGram(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()
        self.center  = nn.Embedding(vocab_size, emb_dim)
        self.context = nn.Embedding(vocab_size, emb_dim)
        nn.init.xavier_uniform_(self.center.weight)
        nn.init.xavier_uniform_(self.context.weight)

    def forward(self, center, pos_ctx, neg_ctx):
        """
        center  : [B]
        pos_ctx : [B]
        neg_ctx : [B, K]
        """
        c_emb   = self.center(center)               # [B, D]
        pos_emb = self.context(pos_ctx)              # [B, D]
        neg_emb = self.context(neg_ctx)              # [B, K, D]

        pos_score = (c_emb * pos_emb).sum(-1)        # [B]
        neg_score = torch.bmm(neg_emb,
                              c_emb.unsqueeze(-1)).squeeze(-1)  # [B, K]

        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-8).mean()
        neg_loss = -torch.log(torch.sigmoid(-neg_score) + 1e-8).mean()
        return pos_loss + neg_loss


# ─────────────────────────────────────────────────────────────────────────────
# Main Node2Vec class
# ─────────────────────────────────────────────────────────────────────────────

class Node2Vec:
    """
    Lightweight Node2Vec that runs entirely on CPU with PyTorch.

    Parameters
    ----------
    G            : NetworkX Graph
    emb_dim      : embedding dimension
    walk_length  : length of each random walk
    num_walks    : number of walks per node
    window       : skip-gram context window
    p, q         : return / in-out hyperparameters
    n_neg        : number of negative samples per positive
    lr           : learning rate
    """

    def __init__(self,
                 G:           nx.Graph,
                 emb_dim:     int   = 64,
                 walk_length: int   = 20,
                 num_walks:   int   = 5,
                 window:      int   = 5,
                 p:           float = 1.0,
                 q:           float = 1.0,
                 n_neg:       int   = 5,
                 lr:          float = 0.01):
        self.G           = G
        self.emb_dim     = emb_dim
        self.walk_length = walk_length
        self.num_walks   = num_walks
        self.window      = window
        self.p           = p
        self.q           = q
        self.n_neg       = n_neg
        self.lr          = lr

        self.node_list = sorted(G.nodes())
        self.node2id   = {n: i for i, n in enumerate(self.node_list)}
        self.vocab_size = len(self.node_list)

        self._build_aliases()

        self.model = SkipGram(self.vocab_size, emb_dim)
        self.embeddings: dict = {}   # populated after fit()

    # ── alias tables ──────────────────────────────────────────────────────────
    def _build_aliases(self):
        G, p, q = self.G, self.p, self.q
        print("  Building alias tables …")
        self.alias_nodes = {}
        for node in G.nodes():
            nbrs = sorted(G.neighbors(node))
            if not nbrs:
                self.alias_nodes[node] = _alias_setup([1.0])
                continue
            unnorm = [G[node][n].get("weight", 1.0) for n in nbrs]
            total  = sum(unnorm)
            probs  = [x / total for x in unnorm]
            self.alias_nodes[node] = _alias_setup(probs)

        self.alias_edges = {}
        for edge in G.edges():
            self.alias_edges[edge] = _get_alias_edge(G, edge[0], edge[1], p, q)
            if not G.is_directed():
                self.alias_edges[(edge[1], edge[0])] = _get_alias_edge(
                    G, edge[1], edge[0], p, q)

    # ── generate walks ────────────────────────────────────────────────────────
    def _generate_walks(self) -> list:
        walks   = []
        nodes   = list(self.G.nodes())
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for n in nodes:
                w = biased_random_walk(self.G, n,
                                       self.walk_length, self.p, self.q,
                                       self.alias_nodes, self.alias_edges)
                walks.append([self.node2id[x] for x in w])
        return walks

    # ── build (center, context) pairs ─────────────────────────────────────────
    def _build_pairs(self, walks) -> list:
        pairs = []
        for walk in walks:
            for i, center in enumerate(walk):
                start = max(0, i - self.window)
                end   = min(len(walk), i + self.window + 1)
                for j in range(start, end):
                    if j != i:
                        pairs.append((center, walk[j]))
        return pairs

    # ── fit ───────────────────────────────────────────────────────────────────
    def fit(self, epochs: int = 3, batch_size: int = 512) -> "Node2Vec":
        print("  Generating random walks …")
        walks = self._generate_walks()
        pairs = self._build_pairs(walks)
        print(f"  Training skip-gram on {len(pairs):,} pairs …")

        node_ids = list(range(self.vocab_size))
        opt      = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(1, epochs + 1):
            random.shuffle(pairs)
            total_loss = 0.0
            n_batches  = 0

            for start in range(0, len(pairs), batch_size):
                batch = pairs[start: start + batch_size]
                centers  = torch.tensor([p[0] for p in batch], dtype=torch.long)
                contexts = torch.tensor([p[1] for p in batch], dtype=torch.long)
                negs     = torch.tensor(
                    np.random.choice(node_ids,
                                     size=(len(batch), self.n_neg)),
                    dtype=torch.long)

                opt.zero_grad()
                loss = self.model(centers, contexts, negs)
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_batches  += 1

            avg_loss = total_loss / max(n_batches, 1)
            print(f"    Node2Vec epoch {epoch}/{epochs} | loss={avg_loss:.4f}")

        # Extract embeddings dict  {node_name: np.array[D]}
        with torch.no_grad():
            emb_matrix = self.model.center.weight.numpy()
        self.embeddings = {n: emb_matrix[self.node2id[n]]
                           for n in self.node_list}
        return self

    # ── wv-style accessor for compatibility ───────────────────────────────────
    def get_vector(self, node) -> np.ndarray:
        return self.embeddings[node]
