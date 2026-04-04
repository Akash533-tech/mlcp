# 🎬 CineGraph — Graph-Based Movie Recommendation System

An end-to-end **Graph Neural Network (GNN)** recommendation system built on the MovieLens-small dataset. Users, movies, and genres are modelled as graph nodes; ratings and interactions as typed edges — enabling higher-order collaborative signals far beyond standard matrix factorisation.

---

## Architecture

```
Nodes
├── 👤 User  (610)    — learnable 64-dim embeddings
├── 🎥 Movie (9,724)  — multi-hot genre + normalised year
└── 🏷️  Genre (19)    — identity embedding

Edges
├── User  ──[rated]──────▶  Movie   (weight = rating × time-decay)
├── Movie ──[belongs_to]──▶ Genre
├── User  ──[similar_to]──  User    (cosine K-NN, K=10)
└── Movie ──[co_watched]──  Movie   (co-occurrence ≥ 5 shared users)
```

**Model:** [LightGCN](https://arxiv.org/abs/2002.02126) — 3 graph convolution layers, BPR loss, mean-pooled final embeddings.  
**Pre-training:** Custom Node2Vec (no gensim dependency) warm-starts the embedding table.  
**Split:** Temporal 70 / 10 / 20 — no data leakage.

---

## Project Structure

```
mlcp/
├── ml-latest-small/          ← MovieLens dataset
│   ├── ratings.csv
│   ├── movies.csv
│   ├── tags.csv
│   └── links.csv
│
├── src/
│   ├── data_loader.py        ← load, preprocess, temporal split
│   ├── graph_builder.py      ← NetworkX + PyG HeteroData
│   ├── node2vec_lite.py      ← custom Node2Vec (pure PyTorch)
│   ├── model.py              ← LightGCN + GraphSAGE + BPR loss
│   ├── train.py              ← training loop, BPR, early stopping
│   ├── evaluate.py           ← P/R/NDCG@K, RMSE, MAE, SVD baseline
│   └── visualize.py          ← matplotlib + PyVis helpers
│
├── gnn_recsys.ipynb          ← 📓 Full pipeline notebook (8 stages)
├── run_pipeline.py           ← 🚀 CLI end-to-end runner
├── app.py                    ← 🌐 Streamlit demo
├── requirements.txt
└── README.md
```

**Generated after training:**
```
model_weights.pth             ← best LightGCN checkpoint
training_history.pt           ← loss/metric curves
eval_results.pt               ← comparison table data
graph_vis.html                ← interactive PyVis graph
eda_overview.png              ← EDA plots
training_curves.png           ← loss + Recall@10 + NDCG@10
tsne_embeddings.png           ← t-SNE of learned embeddings
model_comparison.png          ← final bar chart comparison
```

---

## Quick Start

### 1 — Set up environment

```bash
cd mlcp
python3 -m venv venv
source venv/bin/activate
pip install torch torch-geometric pandas networkx pyvis matplotlib seaborn \
            streamlit tqdm scipy plotly scikit-learn
```

### 2 — Train the model (CLI)

```bash
source venv/bin/activate
python run_pipeline.py --epochs 50 --emb-dim 64 --layers 3
```

Optional flags:
| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 50 | Max training epochs |
| `--emb-dim` | 64 | Embedding dimension |
| `--layers` | 3 | LightGCN conv layers |
| `--lr` | 0.001 | Adam learning rate |
| `--batch` | 1024 | Batch size |
| `--device` | cpu | `cpu` / `cuda` / `mps` |
| `--n2v-walks` | 5 | Node2Vec walks (0 = skip) |

### 3 — Launch the Streamlit demo

```bash
source venv/bin/activate
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### 4 — Run the Jupyter notebook

```bash
source venv/bin/activate
jupyter notebook gnn_recsys.ipynb
```

---

## Pipeline Stages

| # | Stage | Key Output |
|---|-------|-----------|
| 1 | **Data Loading & EDA** | sparsity stats, rating/genre distributions |
| 2 | **Graph Construction** | NetworkX graph (11K+ nodes), PyG HeteroData |
| 3 | **Node2Vec Pre-training** | 64-dim node embeddings via skip-gram |
| 4 | **LightGCN Definition** | 661K-param model, dot-product scoring |
| 5 | **Training** | BPR loss, temporal split, early stopping on Recall@10 |
| 6 | **Evaluation** | Precision/Recall/NDCG@10, RMSE, MAE, Coverage |
| 7 | **Explainability** | similar-user influence tracing, t-SNE embedding viz |
| 8 | **Comparison** | Random vs SVD vs LightGCN table + bar chart |

---

## Evaluation Metrics

| Metric | Measures |
|--------|----------|
| RMSE / MAE | Rating prediction accuracy |
| Precision@10 | Fraction of recommended movies the user actually liked |
| Recall@10 | Fraction of liked movies captured in Top-10 |
| NDCG@10 | Ranking quality (position-aware) |
| Coverage | Fraction of catalogue recommended at least once |

---

## Streamlit Demo — 5 Pages

| Page | Description |
|------|-------------|
| 🏠 Overview | Dataset stats, graph architecture diagram, rating distribution |
| 🎯 Recommendations | Select user → LightGCN Top-10 vs SVD Top-10 side-by-side |
| 📊 EDA & Graph Stats | Genre frequency, ratings/user histogram, activity over time |
| 🔬 Model Training | Live training loss + Recall@10 + NDCG@10 curves |
| ⚖️ Model Comparison | RMSE bar chart + full metrics table |

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Graph Construction | NetworkX, PyTorch Geometric |
| GNN Model | LightGCN (PyG `LGConv`) |
| Node Embeddings | Custom Node2Vec (pure PyTorch) |
| Training | PyTorch, Adam, BPR Loss |
| Evaluation | scikit-learn, scipy SVD |
| Visualisation | Matplotlib, PyVis, Plotly |
| Demo UI | Streamlit |
| Dataset | [MovieLens-small](https://grouplens.org/datasets/movielens/) |

---

## References

- He et al. (2020) — [LightGCN: Simplifying and Powering GCN for Recommendation](https://arxiv.org/abs/2002.02126)  
- Grover & Leskovec (2016) — [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653)  
- Rendle et al. (2009) — [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618)  
- Harper & Konstan (2015) — [The MovieLens Datasets](https://doi.org/10.1145/2827872)
