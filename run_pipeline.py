"""
run_pipeline.py
===============
End-to-end pipeline script.  Run this once to:
  1. Load & pre-process MovieLens data
  2. Build the heterogeneous graph (PyG HeteroData + NetworkX)
  3. Pre-train Node2Vec embeddings (lightweight, no gensim)
  4. Train LightGCN (BPR loss, early stopping)
  5. Evaluate on test set
  6. Run SVD baseline & generate comparison table
  7. Save: model_weights.pth, training_history.pt, eval_results.pt, graph_vis.html

Usage
-----
    source venv/bin/activate
    python run_pipeline.py [--epochs 50] [--emb-dim 64] [--layers 3] [--device cpu]
"""

import argparse, os, sys, time
import torch
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GNN Movie Recommender Pipeline")
parser.add_argument("--epochs",   type=int,   default=50)
parser.add_argument("--emb-dim",  type=int,   default=64)
parser.add_argument("--layers",   type=int,   default=3)
parser.add_argument("--lr",       type=float, default=1e-3)
parser.add_argument("--batch",    type=int,   default=1024)
parser.add_argument("--device",   type=str,   default="cpu",
                    help="cpu | cuda | mps")
parser.add_argument("--n2v-walks",type=int,   default=0,
                    help="Node2Vec walks per node (0 to skip — recommended for large graphs)")
args = parser.parse_args()

DEVICE = args.device
BASE   = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("  STAGE 1 — Data Loading & Preprocessing")
print("═"*60)

from src.data_loader   import load_raw_data, preprocess, temporal_split, sparsity_report
from src.graph_builder import build_networkx_graph, build_pyg_data
from src.model         import LightGCN
from src.train         import Trainer
from src.evaluate      import full_evaluation, baseline_mf, comparison_table

t0  = time.time()
raw  = load_raw_data()
data = preprocess(raw)

train_df, val_df, test_df = temporal_split(data["ratings"])

stats = sparsity_report(data["ratings"], data["n_users"], data["n_movies"])
print(f"  Users   : {stats['n_users']:>6,}")
print(f"  Movies  : {stats['n_movies']:>6,}")
print(f"  Ratings : {stats['n_ratings']:>8,}")
print(f"  Sparsity: {stats['sparsity']*100:.2f}%")
print(f"  Cold-start users (<5 ratings): {len(data['cold_start_users'])}")
print(f"  Train/Val/Test: {len(train_df):,} / {len(val_df):,} / {len(test_df):,}")
print(f"  Done in {time.time()-t0:.1f}s\n")

# ─────────────────────────────────────────────────────────────────────────────
print("═"*60)
print("  STAGE 2 — Graph Construction")
print("═"*60)
t1 = time.time()

pyg = build_pyg_data(data)
print(f"  PyG HeteroData built: {pyg}")

# Only build NetworkX graph if Node2Vec is requested (it's slow on 1.4M edges)
from src.visualize import visualize_graph_pyvis
if args.n2v_walks > 0:
    print("  Building NetworkX graph (for Node2Vec + visualisation) …")
    G = build_networkx_graph(data)
    html_path = os.path.join(BASE, "graph_vis.html")
    visualize_graph_pyvis(G, output_html=html_path, max_nodes=300, max_edges=800)
else:
    print("  Skipping full NetworkX graph build (--n2v-walks 0).")
    print("  Building lightweight subgraph for visualisation only …")
    import networkx as nx
    G = nx.Graph()
    # Add just user-movie edges (first 5000 ratings) for visualisation
    for _, row in data['ratings'].head(5000).iterrows():
        G.add_node(f"u_{int(row['user_idx'])}",  node_type='user')
        G.add_node(f"m_{int(row['movie_idx'])}", node_type='movie')
        G.add_edge(f"u_{int(row['user_idx'])}",
                   f"m_{int(row['movie_idx'])}",
                   edge_type='rated', weight=float(row['rating']))
    html_path = os.path.join(BASE, "graph_vis.html")
    visualize_graph_pyvis(G, output_html=html_path, max_nodes=300, max_edges=800)
print(f"  Done in {time.time()-t1:.1f}s\n")

# ─────────────────────────────────────────────────────────────────────────────
print("═"*60)
print("  STAGE 3 — Node2Vec Pre-training")
print("═"*60)
t2 = time.time()

n2v_embs = None
if args.n2v_walks > 0:
    from src.node2vec_lite import Node2Vec
    # Use only the lightweight subgraph (ratings-only) to avoid alias-table OOM
    n2v = Node2Vec(G,
                   emb_dim=args.emb_dim,
                   walk_length=20,
                   num_walks=args.n2v_walks,
                   window=5,
                   p=1.0, q=1.0,
                   lr=0.02)
    n2v.fit(epochs=3)
    n2v_embs = n2v.embeddings
    print(f"  Node2Vec embeddings: {len(n2v_embs)} nodes × {args.emb_dim} dims")
    print("  (warm-start will be applied to LightGCN)")
else:
    print("  Skipped (--n2v-walks 0)  — LightGCN will use random initialisation.")
print(f"  Done in {time.time()-t2:.1f}s\n")

# ─────────────────────────────────────────────────────────────────────────────
print("═"*60)
print("  STAGE 4 & 5 — LightGCN Training")
print("═"*60)
t3 = time.time()

n_users  = data["n_users"]
n_movies = data["n_movies"]

# Build bipartite edge index for LightGCN
# (users: 0..n_users-1, movies: n_users..n_users+n_movies-1)
ei          = pyg["user", "rated", "movie"].edge_index   # [2, E]
user_src    = ei[0]
movie_dst   = ei[1] + n_users
edge_index  = torch.stack([
    torch.cat([user_src, movie_dst]),
    torch.cat([movie_dst, user_src]),
])  # bidirectional  [2, 2E]

model = LightGCN(n_users=n_users, n_movies=n_movies,
                 emb_dim=args.emb_dim, n_layers=args.layers,
                 dropout=0.1)

# Optional: warm-start from Node2Vec
if n2v_embs is not None:
    with torch.no_grad():
        for uid, uidx in data["user2idx"].items():
            key = f"u_{uidx}"
            if key in n2v_embs:
                vec = torch.tensor(n2v_embs[key], dtype=torch.float)
                model.user_emb.weight[uidx] = vec
        for mid, midx in data["movie2idx"].items():
            key = f"m_{midx}"
            if key in n2v_embs:
                vec = torch.tensor(n2v_embs[key], dtype=torch.float)
                model.movie_emb.weight[midx] = vec
    print("  Warm-started LightGCN from Node2Vec embeddings")

trainer = Trainer(
    model=model,
    edge_index=edge_index,
    train_df=train_df,
    val_df=val_df,
    device=DEVICE,
    lr=args.lr,
    batch_size=args.batch,
    l2_lambda=1e-4,
)

WEIGHTS_PATH = os.path.join(BASE, "model_weights.pth")
history = trainer.fit(
    epochs=args.epochs,
    patience=7,
    eval_every=5,
    save_path=WEIGHTS_PATH,
)

# Save training history
HISTORY_PATH = os.path.join(BASE, "training_history.pt")
torch.save(history, HISTORY_PATH)
print(f"\n  Training history saved → {HISTORY_PATH}")
print(f"  Done in {time.time()-t3:.1f}s\n")

# ─────────────────────────────────────────────────────────────────────────────
print("═"*60)
print("  STAGE 6 — Evaluation & Comparison")
print("═"*60)
t4 = time.time()

# Load best weights
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))

gnn_metrics = full_evaluation(model, edge_index, test_df, train_df,
                               device=DEVICE, K=10, max_users=500)
print("\n  ── GNN (LightGCN) Test Metrics ──")
for k, v in gnn_metrics.items():
    print(f"    {k:25s}: {v:.4f}" if isinstance(v, float) else f"    {k:25s}: {v}")

print("\n  ── SVD Baseline ──")
svd_metrics = baseline_mf(train_df, test_df, n_users, n_movies, k=50)
print(f"    RMSE: {svd_metrics['rmse']:.4f}")
print(f"    MAE : {svd_metrics['mae']:.4f}")

print("\n  ── Comparison Table ──")
table = comparison_table(gnn_metrics, svd_metrics)
print(table.to_string())

# Save results
RESULTS_PATH = os.path.join(BASE, "eval_results.pt")
torch.save({"gnn": gnn_metrics, "svd": svd_metrics}, RESULTS_PATH)
print(f"\n  Results saved → {RESULTS_PATH}")
print(f"  Done in {time.time()-t4:.1f}s\n")

# ─────────────────────────────────────────────────────────────────────────────
print("═"*60)
print("  ✅  PIPELINE COMPLETE")
print("═"*60)
print(f"\n  Total wall-clock time : {(time.time()-t0)/60:.1f} min")
print(f"  Artifacts generated:")
print(f"    model_weights.pth      ← trained LightGCN")
print(f"    training_history.pt    ← loss/metric curves")
print(f"    eval_results.pt        ← comparison table data")
print(f"    graph_vis.html         ← interactive graph")
print(f"\n  Launch the Streamlit demo:")
print(f"    streamlit run app.py")
print()
