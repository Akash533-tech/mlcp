"""
src/train.py
============
Training loop and evaluation harness for LightGCN / GraphSAGE.

Usage
-----
    from src.train import Trainer
    trainer = Trainer(model, edge_index, train_df, val_df, device)
    history = trainer.fit(epochs=50)
"""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from collections import defaultdict

from src.model import bpr_loss, mse_loss


# ─────────────────────────────────────────────────────────────────────────────
class RatingDataset(Dataset):
    """Positive (user, movie, rating) triples from a DataFrame."""

    def __init__(self, df, n_movies):
        self.users   = torch.tensor(df["user_idx"].values,  dtype=torch.long)
        self.movies  = torch.tensor(df["movie_idx"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values,    dtype=torch.float)
        self.n_movies = n_movies

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        # Sample a negative movie uniformly
        neg = torch.randint(0, self.n_movies, (1,)).item()
        return (self.users[idx], self.movies[idx],
                torch.tensor(neg, dtype=torch.long),
                self.ratings[idx])


# ─────────────────────────────────────────────────────────────────────────────
class Trainer:
    def __init__(self,
                 model,
                 edge_index:  torch.Tensor,
                 train_df,
                 val_df,
                 device:      str  = "cpu",
                 lr:          float = 1e-3,
                 batch_size:  int   = 1024,
                 l2_lambda:   float = 1e-4,
                 edge_weight: torch.Tensor = None):
        self.model       = model.to(device)
        self.edge_index  = edge_index.to(device)
        self.edge_weight = edge_weight.to(device) if edge_weight is not None else None
        self.train_df    = train_df
        self.val_df      = val_df
        self.device      = device
        self.l2_lambda   = l2_lambda
        self.batch_size  = batch_size

        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        n_movies = model.n_movies
        self.train_ds = RatingDataset(train_df, n_movies)
        self.train_loader = DataLoader(
            self.train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=(device != "cpu"))

        # Ground-truth sets for ranking evaluation
        self._build_ground_truth()

    # ── build ground truth ────────────────────────────────────────────────────
    def _build_ground_truth(self):
        """Build {user_idx: set(movie_idx)} for train and val."""
        self.train_user_items = defaultdict(set)
        for _, row in self.train_df.iterrows():
            self.train_user_items[int(row["user_idx"])].add(int(row["movie_idx"]))

        self.val_user_items = defaultdict(set)
        for _, row in self.val_df.iterrows():
            self.val_user_items[int(row["user_idx"])].add(int(row["movie_idx"]))

    # ── one training epoch ────────────────────────────────────────────────────
    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            u, pos, neg, rating = [b.to(self.device) for b in batch]
            self.optimizer.zero_grad()

            user_embs, movie_embs = self.model(self.edge_index, self.edge_weight)

            u_e   = user_embs[u]
            pos_e = movie_embs[pos]
            neg_e = movie_embs[neg]

            loss = bpr_loss(u_e, pos_e, neg_e, self.l2_lambda)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(u)

        return total_loss / len(self.train_ds)

    # ── ranking metrics ───────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self, K: int = 10) -> dict:
        self.model.eval()
        user_embs, movie_embs = self.model(self.edge_index, self.edge_weight)

        precisions, recalls, ndcgs, mses = [], [], [], []

        eval_users = list(self.val_user_items.keys())[:200]  # cap for speed

        for u in eval_users:
            gt = self.val_user_items[u]
            if not gt:
                continue
            exclude = self.train_user_items[u]

            scores = (movie_embs @ user_embs[u])  # [n_movies]
            for m in exclude:
                scores[m] = float("-inf")

            top_k = scores.topk(K).indices.cpu().numpy()

            # Precision & Recall
            hits = sum(1 for m in top_k if m in gt)
            precisions.append(hits / K)
            recalls.append(hits / len(gt))

            # NDCG
            dcg  = sum((1 / np.log2(i + 2)) for i, m in enumerate(top_k) if m in gt)
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(gt), K)))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

            # MSE on val interactions for this user
            val_m = torch.tensor(list(gt), device=self.device)
            true_r = self.val_df[
                (self.val_df["user_idx"] == u) &
                (self.val_df["movie_idx"].isin(gt))
            ]["rating"].values
            if len(true_r) > 0:
                pred_r = (movie_embs[val_m] * user_embs[u]).sum(-1).cpu().numpy()
                pred_r = np.clip(pred_r, 0.5, 5.0)
                mse_val = np.mean((pred_r[:len(true_r)] - true_r) ** 2)
                mses.append(mse_val)

        return dict(
            precision_at_k=float(np.mean(precisions)),
            recall_at_k=float(np.mean(recalls)),
            ndcg_at_k=float(np.mean(ndcgs)),
            rmse=float(np.sqrt(np.mean(mses))) if mses else float("nan"),
            mae=float(np.mean([np.sqrt(x) for x in mses])) if mses else float("nan"),
        )

    # ── main fit loop ─────────────────────────────────────────────────────────
    def fit(self,
            epochs:        int   = 50,
            patience:      int   = 5,
            eval_every:    int   = 5,
            save_path:     str   = "model_weights.pth") -> list:
        """
        Train for `epochs` steps with early stopping on Recall@K.

        Returns
        -------
        history : list of dicts (one per evaluation step)
        """
        history   = []
        best_recall = 0.0
        no_improve  = 0

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch()

            if epoch % eval_every == 0 or epoch == epochs:
                metrics = self.evaluate(K=10)
                metrics["epoch"]      = epoch
                metrics["train_loss"] = train_loss
                history.append(metrics)
                print(f"Epoch {epoch:3d} | loss={train_loss:.4f} | "
                      f"P@10={metrics['precision_at_k']:.4f} | "
                      f"R@10={metrics['recall_at_k']:.4f} | "
                      f"NDCG@10={metrics['ndcg_at_k']:.4f} | "
                      f"RMSE={metrics['rmse']:.4f}")

                if metrics["recall_at_k"] > best_recall:
                    best_recall = metrics["recall_at_k"]
                    no_improve  = 0
                    torch.save(self.model.state_dict(), save_path)
                    print(f"  ✓ Best model saved (Recall@10={best_recall:.4f})")
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"  Early stopping at epoch {epoch}")
                        break
            else:
                print(f"Epoch {epoch:3d} | loss={train_loss:.4f}")

        return history
