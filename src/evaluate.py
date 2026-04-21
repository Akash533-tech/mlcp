"""
src/evaluate.py
===============
Offline evaluation utilities.

Provides:
  - hit_at_k, precision_at_k, recall_at_k, ndcg_at_k
  - full_evaluation()  – runs all metrics on a test set
  - baseline_mf()      – trains a simple SVD (via scipy) and computes RMSE/MAE
  - comparison_table() – dict comparing MF vs GNN results
"""

import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Pointwise metrics (ranking)
# ─────────────────────────────────────────────────────────────────────────────

def ndcg_at_k(recommended: list, ground_truth: set, k: int) -> float:
    dcg  = sum(1 / np.log2(i + 2) for i, m in enumerate(recommended[:k]) if m in ground_truth)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(recommended: list, ground_truth: set, k: int) -> float:
    hits = sum(1 for m in recommended[:k] if m in ground_truth)
    return hits / k


def recall_at_k(recommended: list, ground_truth: set, k: int) -> float:
    hits = sum(1 for m in recommended[:k] if m in ground_truth)
    return hits / len(ground_truth) if ground_truth else 0.0


# ─────────────────────────────────────────────────────────────────────────────
def full_evaluation(model,
                    edge_index:   torch.Tensor,
                    test_df:      pd.DataFrame,
                    train_df:     pd.DataFrame,
                    device:       str = "cpu",
                    K:            int = 10,
                    max_users:    int = 500,
                    edge_weight:  torch.Tensor = None) -> dict:
    """
    Compute Precision@K, Recall@K, NDCG@K, RMSE, MAE on the test split.
    Also computes Coverage and Diversity.
    """
    model.eval()
    model.to(device)
    edge_index = edge_index.to(device)

    train_user_items = defaultdict(set)
    for _, row in train_df.iterrows():
        train_user_items[int(row["user_idx"])].add(int(row["movie_idx"]))

    test_user_items = defaultdict(set)
    test_user_ratings = defaultdict(list)
    for _, row in test_df.iterrows():
        uid = int(row["user_idx"])
        mid = int(row["movie_idx"])
        test_user_items[uid].add(mid)
        test_user_ratings[uid].append((mid, float(row["rating"])))

    eval_users = list(test_user_items.keys())[:max_users]

    precisions, recalls, ndcgs = [], [], []
    all_rmse, all_mae           = [], []
    all_recommended             = []

    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    with torch.no_grad():
        user_embs, movie_embs = model(edge_index, edge_weight)

    for u in eval_users:
        gt = test_user_items[u]
        if not gt:
            continue
        exclude = train_user_items[u]

        scores = (movie_embs @ user_embs[u]).cpu()
        for m in exclude:
            scores[m] = float("-inf")

        top_k = scores.topk(K).indices.tolist()
        all_recommended.extend(top_k)

        precisions.append(precision_at_k(top_k, gt, K))
        recalls.append(recall_at_k(top_k, gt, K))
        ndcgs.append(ndcg_at_k(top_k, gt, K))

        # RMSE / MAE – predict ratings for test interactions
        gt_list = list(gt)
        gt_t    = torch.tensor(gt_list, device=device)
        pred_r  = (movie_embs[gt_t] * user_embs[u]).sum(-1).cpu().numpy()
        pred_r  = np.clip(pred_r, 0.5, 5.0)
        true_r  = np.array([dict(test_user_ratings[u]).get(m, 3.0) for m in gt_list])
        all_rmse.append(np.sqrt(np.mean((pred_r - true_r) ** 2)))
        all_mae.append(np.mean(np.abs(pred_r - true_r)))

    # Coverage = fraction of catalogue recommended at least once
    n_movies  = model.n_movies
    coverage  = len(set(all_recommended)) / n_movies

    # Diversity = average pairwise distance (crude: 1 - overlap of recommendations)
    # We approximate via intra-list diversity per user (mean of pair similarities)
    with torch.no_grad():
        norm_embs = F.normalize(movie_embs, dim=-1) if hasattr(F, 'normalize') \
                    else movie_embs / (movie_embs.norm(dim=-1, keepdim=True) + 1e-8)

    return dict(
        precision_at_k=float(np.mean(precisions)),
        recall_at_k=float(np.mean(recalls)),
        ndcg_at_k=float(np.mean(ndcgs)),
        rmse=float(np.mean(all_rmse)),
        mae=float(np.mean(all_mae)),
        coverage=float(coverage),
        n_users_evaluated=len(eval_users),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Baseline Matrix Factorisation (SVD)
# ─────────────────────────────────────────────────────────────────────────────

def baseline_mf(train_df: pd.DataFrame,
                test_df:  pd.DataFrame,
                n_users:  int,
                n_movies: int,
                k:        int = 50) -> dict:
    """
    Fit a truncated SVD (k components) on the explicit rating matrix
    and evaluate RMSE / MAE on the test set.
    """
    # Build rating matrix (mean-normalised)
    row = train_df["user_idx"].values.astype(int)
    col = train_df["movie_idx"].values.astype(int)
    val = train_df["rating"].values.astype(float)

    R = csr_matrix((val, (row, col)), shape=(n_users, n_movies)).toarray()
    user_means = np.where(R > 0, R, np.nan)
    user_means = np.nanmean(user_means, axis=1, keepdims=True)
    user_means = np.nan_to_num(user_means, nan=0.0)
    R_norm = R - user_means * (R > 0)

    k = min(k, min(n_users, n_movies) - 1)
    U, sigma, Vt = svds(R_norm, k=k)
    R_pred = U @ np.diag(sigma) @ Vt + user_means

    # Evaluate
    rmse_list, mae_list = [], []
    for _, row_r in test_df.iterrows():
        u = int(row_r["user_idx"])
        m = int(row_r["movie_idx"])
        pred = np.clip(R_pred[u, m], 0.5, 5.0)
        true = float(row_r["rating"])
        rmse_list.append((pred - true) ** 2)
        mae_list.append(abs(pred - true))

    return dict(
        model="SVD (k=50)",
        rmse=float(np.sqrt(np.mean(rmse_list))),
        mae=float(np.mean(mae_list)),
    )


# ─────────────────────────────────────────────────────────────────────────────
def comparison_table(gnn_metrics: dict,
                     svd_metrics: dict) -> pd.DataFrame:
    """Return a formatted DataFrame comparing models."""
    rows = [
        {"Model": "Matrix Factorisation (SVD)",
         "RMSE":           round(svd_metrics["rmse"], 4),
         "MAE":            round(svd_metrics["mae"],  4),
         "Precision@10":   "N/A",
         "Recall@10":      "N/A",
         "NDCG@10":        "N/A",
         "Coverage":       "N/A"},
        {"Model": "LightGCN (GNN)",
         "RMSE":           round(gnn_metrics["rmse"],            4),
         "MAE":            round(gnn_metrics["mae"],             4),
         "Precision@10":   round(gnn_metrics["precision_at_k"],  4),
         "Recall@10":      round(gnn_metrics["recall_at_k"],     4),
         "NDCG@10":        round(gnn_metrics["ndcg_at_k"],       4),
         "Coverage":       round(gnn_metrics["coverage"],        4)},
    ]
    return pd.DataFrame(rows).set_index("Model")


# ─────────────────────────────────────────────────────────────────────────────
# keep F importable even if torch.nn.functional not yet imported at top level
import torch.nn.functional as F
