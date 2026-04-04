"""
src/data_loader.py
==================
Loads and pre-processes the MovieLens small dataset.
Handles cold-start (< 5 ratings), sparsity analysis, and
normalisation helpers used throughout the pipeline.
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml-latest-small")


# ─────────────────────────────────────────────────────────────────────────────
def load_raw_data() -> dict:
    """Return a dict of raw DataFrames: ratings, movies, tags, links."""
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    movies  = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    tags    = pd.read_csv(os.path.join(DATA_DIR, "tags.csv"))
    links   = pd.read_csv(os.path.join(DATA_DIR, "links.csv"))
    return dict(ratings=ratings, movies=movies, tags=tags, links=links)


# ─────────────────────────────────────────────────────────────────────────────
def extract_year(title: str) -> int:
    """Extract the release year embedded in a MovieLens title string."""
    import re
    m = re.search(r"\((\d{4})\)", title)
    return int(m.group(1)) if m else 0


# ─────────────────────────────────────────────────────────────────────────────
def preprocess(raw: dict, cold_start_threshold: int = 5) -> dict:
    """
    Clean and enrich raw data.

    Returns
    -------
    dict with keys:
        ratings   – cleaned ratings DataFrame (userId, movieId, rating, timestamp)
        movies    – enriched movies DataFrame  (movieId, title, year, genres list)
        tags      – tags DataFrame
        links     – links DataFrame
        genre_list – sorted list of all unique genre strings
        cold_start_users – userId series with < cold_start_threshold ratings
    """
    ratings = raw["ratings"].copy()
    movies  = raw["movies"].copy()
    tags    = raw["tags"].copy()
    links   = raw["links"].copy()

    # ── Movie enrichment ──────────────────────────────────────────────────────
    movies["year"]   = movies["title"].apply(extract_year)
    movies["genres"] = movies["genres"].apply(
        lambda g: [] if g == "(no genres listed)" else g.split("|")
    )

    # ── Cold-start identification ─────────────────────────────────────────────
    rating_counts    = ratings.groupby("userId").size()
    cold_start_users = rating_counts[rating_counts < cold_start_threshold].index

    # ── Consecutive integer indices for PyTorch ───────────────────────────────
    unique_users  = sorted(ratings["userId"].unique())
    unique_movies = sorted(ratings["movieId"].unique())

    user2idx  = {u: i for i, u in enumerate(unique_users)}
    movie2idx = {m: i for i, m in enumerate(unique_movies)}

    ratings["user_idx"]  = ratings["userId"].map(user2idx)
    ratings["movie_idx"] = ratings["movieId"].map(movie2idx)

    # ── Genre master list ─────────────────────────────────────────────────────
    genre_list = sorted({g for genres in movies["genres"] for g in genres})

    return dict(
        ratings=ratings,
        movies=movies,
        tags=tags,
        links=links,
        user2idx=user2idx,
        movie2idx=movie2idx,
        genre_list=genre_list,
        cold_start_users=cold_start_users,
        n_users=len(unique_users),
        n_movies=len(unique_movies),
    )


# ─────────────────────────────────────────────────────────────────────────────
def temporal_split(ratings: pd.DataFrame,
                   val_frac: float = 0.10,
                   test_frac: float = 0.20
                   ) -> tuple:
    """
    Temporal train/val/test split.
    The most recent (test_frac) interactions go to test,
    the next (val_frac) go to val, the rest go to train.
    """
    ratings_sorted = ratings.sort_values("timestamp")
    n = len(ratings_sorted)
    test_start = int(n * (1 - test_frac))
    val_start  = int(n * (1 - test_frac - val_frac))

    train = ratings_sorted.iloc[:val_start]
    val   = ratings_sorted.iloc[val_start:test_start]
    test  = ratings_sorted.iloc[test_start:]
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
def sparsity_report(ratings: pd.DataFrame,
                    n_users: int,
                    n_movies: int) -> dict:
    """Return a dict of basic sparsity statistics."""
    total_possible = n_users * n_movies
    observed       = len(ratings)
    sparsity       = 1.0 - observed / total_possible
    return dict(
        n_users=n_users,
        n_movies=n_movies,
        n_ratings=observed,
        sparsity=sparsity,
        avg_ratings_per_user=observed / n_users,
        avg_ratings_per_movie=observed / n_movies,
    )
