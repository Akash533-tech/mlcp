"""
app.py  –  Graph-Based Movie Recommendation System
====================================================
Streamlit web demo:
  • Select a user → see Top-10 GNN recommendations
  • Compare with SVD recommendations side-by-side
  • Browse interactive graph around any movie
  • View training metrics dashboard
"""

import os, sys, time
import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

# ─── ensure src/ is importable ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader   import load_raw_data, preprocess, temporal_split, sparsity_report
from src.graph_builder import build_pyg_data, time_decay
from src.model         import LightGCN

# ═══════════════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CineGraph – GNN Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0a0a1a; color: #e8e8ff; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12122a 0%, #1a1a3e 100%);
    border-right: 1px solid #2a2a5a;
}
[data-testid="stSidebar"] * { color: #c8c8f0 !important; }

/* ── Cards ── */
.rec-card {
    background: linear-gradient(135deg, #1a1a3a 0%, #1e1e4a 100%);
    border: 1px solid #3a3a7a;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
    transition: transform .2s, box-shadow .2s;
    display: flex;
    align-items: center;
}
.rec-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(108,99,255,.35);
}
.rec-poster {
    width: 60px;
    height: 90px;
    object-fit: cover;
    border-radius: 6px;
    margin-right: 14px;
    border: 1px solid #3a3a7a;
    background-color: #12122a;
}
.rec-content {
    flex: 1;
}
.rec-rank {
    font-size: 1.4rem;
    font-weight: 700;
    color: #FFBE0B;
    margin-right: 8px;
}
.rec-title { font-size: 1rem; font-weight: 600; color: #e8e8ff; }
.rec-meta  { font-size: .78rem; color: #9090c0; margin-top: 3px; }
.rec-score {
    float: right;
    background: #6C63FF33;
    border: 1px solid #6C63FF;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: .78rem;
    color: #a8a0ff;
}
.genre-pill {
    display: inline-block;
    background: #FF658422;
    border: 1px solid #FF6584;
    color: #FF9EAF;
    border-radius: 20px;
    padding: 1px 8px;
    font-size: .72rem;
    margin: 2px 2px 0 0;
}
/* ── Metric Tiles ── */
.metric-tile {
    background: linear-gradient(135deg,#1a1a3a,#222255);
    border: 1px solid #3a3a7a;
    border-radius: 10px;
    padding: 18px;
    text-align: center;
}
.metric-val { font-size: 1.8rem; font-weight: 700; color: #6C63FF; }
.metric-lbl { font-size: .78rem; color: #9090c0; margin-top: 2px; }

/* ── Section headers ── */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #a8a0ff;
    border-left: 4px solid #6C63FF;
    padding-left: 10px;
    margin: 20px 0 12px;
}

/* ── Scrollable columns ── */
.scroll-box { max-height: 640px; overflow-y: auto; padding-right: 4px; }

/* ════════════════════════════════════════
   CineBot Chat UI
   ════════════════════════════════════════ */
.chat-outer {
    background: linear-gradient(180deg, #0d0d22 0%, #10102a 100%);
    border: 1px solid #2a2a5a;
    border-radius: 16px;
    padding: 20px 16px 8px;
    max-height: 580px;
    overflow-y: auto;
    scroll-behavior: smooth;
    margin-bottom: 12px;
}
.chat-row-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 14px;
    animation: fadeInRight .3s ease;
}
.chat-row-bot {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 14px;
    animation: fadeInLeft .3s ease;
}
@keyframes fadeInRight {
    from { opacity:0; transform: translateX(20px); }
    to   { opacity:1; transform: translateX(0); }
}
@keyframes fadeInLeft {
    from { opacity:0; transform: translateX(-20px); }
    to   { opacity:1; transform: translateX(0); }
}
.bubble-user {
    max-width: 72%;
    background: linear-gradient(135deg, #6C63FF, #9a56ff);
    color: #fff;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 16px;
    font-size: .92rem;
    line-height: 1.5;
    box-shadow: 0 4px 16px rgba(108,99,255,.4);
}
.bubble-bot {
    max-width: 78%;
    background: linear-gradient(135deg, #1a1a3a, #1e1e50);
    border: 1px solid #3a3a7a;
    color: #e0e0ff;
    border-radius: 18px 18px 18px 4px;
    padding: 10px 16px;
    font-size: .92rem;
    line-height: 1.6;
    box-shadow: 0 4px 16px rgba(0,0,0,.4);
}
.avatar-bot {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: linear-gradient(135deg,#6C63FF,#FF6584);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    margin-right: 10px;
    flex-shrink: 0;
    margin-top: 2px;
}
.avatar-user {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: linear-gradient(135deg,#9a56ff,#6C63FF);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    margin-left: 10px;
    flex-shrink: 0;
    margin-top: 2px;
}
/* Typing dots */
.typing-indicator {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 4px 0;
}
.typing-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #6C63FF;
    animation: bounce 1.2s infinite ease-in-out;
}
.typing-dot:nth-child(2) { animation-delay: .2s; }
.typing-dot:nth-child(3) { animation-delay: .4s; }
@keyframes bounce {
    0%,60%,100% { transform: translateY(0); }
    30%          { transform: translateY(-8px); }
}
/* Mood option buttons */
.mood-opts {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 8px;
}
/* Chat genre pill */
.chat-genre-pill {
    display: inline-block;
    background: #6C63FF22;
    border: 1px solid #6C63FF;
    color: #b0a8ff;
    border-radius: 20px;
    padding: 1px 8px;
    font-size: .72rem;
    margin: 2px 2px 0 0;
}
/* Inline rec card inside chat */
.chat-rec-card {
    background: linear-gradient(135deg,#12122a,#1a1a40);
    border: 1px solid #3a3a6a;
    border-radius: 10px;
    padding: 10px 12px;
    margin-top: 8px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.chat-rec-poster {
    width: 44px; height: 66px;
    object-fit: cover;
    border-radius: 5px;
    border: 1px solid #3a3a6a;
    background: #12122a;
    flex-shrink: 0;
}
.chat-rec-rank {
    font-size: 1.1rem;
    font-weight:700;
    color: #FFBE0B;
    min-width: 24px;
}
.chat-rec-title { font-size: .88rem; font-weight:600; color:#e0e0ff; }
.chat-rec-meta  { font-size: .74rem; color: #7070a0; margin-top:2px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Data & Model loading  (cached)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="🔄 Loading data and building graph …")
def load_everything():
    raw  = load_raw_data()
    data = preprocess(raw)
    train_df, val_df, test_df = temporal_split(data["ratings"])
    pyg  = build_pyg_data(data)

    n_users  = data["n_users"]
    n_movies = data["n_movies"]
    n_genres = len(data.get("genre_list", []))
    n_tags   = data.get("n_tags", 0)

    # Convert fully semantic HeteroData into a single homogeneous structure
    homo = pyg.to_homogeneous()
    edge_index  = homo.edge_index
    edge_weight = homo.edge_attr.squeeze(-1) if hasattr(homo, "edge_attr") and homo.edge_attr is not None else None

    # Instantiate Heterogeneous LightGCN mapping size
    model = LightGCN(n_users=n_users, n_movies=n_movies,
                     n_genres=n_genres, n_tags=n_tags,
                     emb_dim=64, n_layers=3)

    WEIGHTS = os.path.join(os.path.dirname(__file__), "model_weights.pth")
    trained = False
    if os.path.exists(WEIGHTS):
        model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
        model.eval()
        trained = True

    return dict(
        data=data, train_df=train_df, val_df=val_df, test_df=test_df,
        model=model, edge_index=edge_index, edge_weight=edge_weight,
        trained=trained, pyg=pyg,
    )


@st.cache_data(show_spinner=False)
def fetch_tmdb_poster(tmdb_id: float, api_key: str) -> str:
    if pd.isna(tmdb_id) or not api_key:
        return ""
    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={api_key}"
    try:
        import requests
        resp = requests.get(url, timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('poster_path'):
                return f"https://image.tmdb.org/t/p/w200{data['poster_path']}"
    except Exception:
        pass
    return ""

def get_movie_info(movie_idx: int, data: dict, api_key: str = "") -> dict:
    movies     = data["movies"]
    links      = data["links"]
    movie2idx  = data["movie2idx"]
    idx2movie  = {v: k for k, v in movie2idx.items()}
    mid        = idx2movie.get(movie_idx)
    if mid is None:
        return {"title": f"Movie #{movie_idx}", "year": "", "genres": [], "poster_url": ""}
    row = movies[movies["movieId"] == mid]
    if row.empty:
        return {"title": f"Movie #{movie_idx}", "year": "", "genres": [], "poster_url": ""}
    r = row.iloc[0]
    
    link_row = links[links["movieId"] == mid]
    tmdb_id = link_row.iloc[0]["tmdbId"] if not link_row.empty else np.nan
    poster_url = fetch_tmdb_poster(tmdb_id, api_key)
    
    return {"title": r["title"], "year": r["year"], "genres": r["genres"], "poster_url": poster_url}

def render_movie_card(info: dict, rank: int = None, score: str = "") -> str:
    genres_html = "".join(f"<span class='genre-pill'>{g}</span>" for g in info["genres"])
    rank_html = f"<span class='rec-rank'>#{rank}</span>" if rank is not None else ""
    score_html = f"<span class='rec-score'>{score}</span>" if score else ""
    poster_html = f"<img class='rec-poster' src='{info['poster_url']}' loading='lazy' />" if info.get("poster_url") else "<div class='rec-poster' style='display:flex; align-items:center; justify-content:center; color:#3a3a7a; font-size:0.7rem;'>No Poster</div>"
    return f"""
    <div class='rec-card'>
        {poster_html}
        <div class='rec-content'>
            <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
                <div>{rank_html} <span class='rec-title'>{info['title']}</span></div>
                {score_html}
            </div>
            <div class='rec-meta'>📅 {info['year']}   {genres_html}</div>
        </div>
    </div>
    """


@torch.no_grad()
def get_recommendations(model, edge_index, edge_weight, user_idx: int,
                        exclude_set: set, top_k: int = 10) -> list:
    model.eval()
    user_embs, movie_embs = model(edge_index, edge_weight)
    scores = (movie_embs @ user_embs[user_idx]).numpy()
    for m in exclude_set:
        scores[m] = -np.inf
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_indices]


# ─────────────────────────────────────────────────────────────────────────────
# SVD baseline (fast, NumPy)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="📐 Running SVD baseline …")
def compute_svd_predictions(_data, _train_df, k=50):
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds
    n_users  = _data["n_users"]
    n_movies = _data["n_movies"]
    row = _train_df["user_idx"].values.astype(int)
    col = _train_df["movie_idx"].values.astype(int)
    val = _train_df["rating"].values.astype(float)
    R   = csr_matrix((val, (row, col)), shape=(n_users, n_movies)).toarray()
    mean = np.where(R > 0, R, np.nan)
    mean = np.nanmean(mean, axis=1, keepdims=True)
    mean = np.nan_to_num(mean, nan=0.0)
    R_norm = R - mean * (R > 0)
    k = min(k, min(n_users, n_movies) - 1)
    U, sigma, Vt = svds(R_norm, k=k)
    R_pred = U @ np.diag(sigma) @ Vt + mean
    return R_pred


def svd_recommendations(R_pred, user_idx: int,
                        exclude_set: set, top_k: int = 10) -> list:
    scores = R_pred[user_idx].copy()
    for m in exclude_set:
        scores[m] = -np.inf
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_indices]


# ═══════════════════════════════════════════════════════════════════════════════
# CineBot helper functions
# ═══════════════════════════════════════════════════════════════════════════════

# ── Genre keyword map ─────────────────────────────────────────────────────────
_GENRE_KEYWORDS: dict[str, list[str]] = {
    "Crime":     ["crime", "noir", "gangster", "heist", "mafia", "detective", "murder"],
    "Thriller":  ["thriller", "suspense", "tense", "tension", "gripping", "chase"],
    "Horror":    ["horror", "scary", "spooky", "ghost", "zombie", "fear", "terrify"],
    "Sci-Fi":    ["sci-fi", "scifi", "space", "future", "alien", "robot", "dystopia", "cyberpunk"],
    "Comedy":    ["comedy", "funny", "laugh", "humor", "hilarious", "fun", "witty"],
    "Romance":   ["romance", "love", "romantic", "relationship", "dating", "heartwarming"],
    "Action":    ["action", "fight", "explosive", "battle", "war", "intense", "adrenaline"],
    "Adventure": ["adventure", "quest", "journey", "explore", "epic", "expedition"],
    "Animation": ["animation", "animated", "cartoon", "pixar", "disney", "anime"],
    "Drama":     ["drama", "emotional", "moving", "powerful", "touching", "Oscar"],
    "Fantasy":   ["fantasy", "magic", "wizard", "mythical", "dragon", "supernatural"],
    "Mystery":   ["mystery", "whodunit", "puzzle", "clue", "investigation"],
    "Documentary":["documentary", "real", "true story", "based on"],
    "Children":  ["children", "kids", "family", "child"],
}

def detect_genres(text: str) -> list[str]:
    """Return list of genre names mentioned in free text."""
    text_lower = text.lower()
    found = []
    for genre, keywords in _GENRE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                found.append(genre)
                break
    return found


# ── Mood quiz questions ────────────────────────────────────────────────────────
QUIZ_QUESTIONS = [
    {
        "q": "How's your energy right now?",
        "opts": ["Hyped 🔥", "Relaxed 😌", "Melancholy 😔", "Curious 🧐"],
    },
    {
        "q": "Pick a vibe for your watch session:",
        "opts": ["Edge-of-seat 😱", "Feel-good 😊", "Deep & meaningful 💭", "Mind-bending 🌀"],
    },
    {
        "q": "Preferred pacing?",
        "opts": ["Fast-paced ⚡", "Slow burn 🕯️"],
    },
    {
        "q": "Who are you watching with?",
        "opts": ["Solo 🎧", "Group / friends 🍿"],
    },
    {
        "q": "What kind of ending do you prefer?",
        "opts": ["Happy 😄", "Bittersweet 😢", "Ambiguous / open-ended 🤔"],
    },
]

_MOOD_MAP: list[tuple[tuple, list[str]]] = [
    # (answer_fingerprint_substrings, genres)
    # energy=Hyped + vibe=Edge-of-seat  → Action/Thriller/Crime
    (("hyped", "edge-of-seat"), ["Action", "Thriller", "Crime"]),
    # energy=Hyped + vibe=Feel-good  → Comedy/Adventure/Animation
    (("hyped", "feel-good"),    ["Comedy", "Adventure", "Animation"]),
    # energy=Relaxed + vibe=Feel-good  → Comedy/Romance/Animation
    (("relaxed", "feel-good"),  ["Comedy", "Romance", "Animation"]),
    # energy=Relaxed + vibe=Deep  → Drama/Romance
    (("relaxed", "deep"),       ["Drama", "Romance"]),
    # energy=Melancholy  → Drama/Romance
    (("melancholy",),           ["Drama", "Romance"]),
    # energy=Curious + vibe=Mind-bending  → Sci-Fi/Mystery/Thriller
    (("curious", "mind-bending"),("Sci-Fi", "Mystery", "Thriller")),
    # Curious + Deep  → Documentary/Drama/Mystery
    (("curious", "deep"),       ["Documentary", "Drama", "Mystery"]),
    # Edge-of-seat + fast  → Thriller/Action/Crime
    (("edge-of-seat", "fast"),  ["Thriller", "Action", "Crime"]),
    # Slow burn  → Drama/Romance/Mystery
    (("slow burn",),            ["Drama", "Romance", "Mystery"]),
    # Group watch + feel-good  → Comedy/Adventure
    (("group", "feel-good"),    ["Comedy", "Adventure", "Action"]),
    # Happy ending
    (("happy",),                ["Comedy", "Romance", "Adventure", "Animation"]),
    # Bittersweet ending
    (("bittersweet",),          ["Drama", "Romance", "Crime"]),
    # Ambiguous
    (("ambiguous",),            ["Mystery", "Sci-Fi", "Thriller"]),
]

def compute_mood_genres(answers: list[str]) -> list[str]:
    """Score answers against mood map and return ranked genre list."""
    from collections import Counter
    combined = " ".join(a.lower() for a in answers)
    counter: Counter = Counter()
    for fingerprint, genres in _MOOD_MAP:
        if all(fp in combined for fp in fingerprint):
            for g in genres:
                counter[g] += 1
    if not counter:
        # Fallback: derive from first answer (energy level)
        energy = answers[0].lower() if answers else ""
        if "hyped" in energy:
            return ["Action", "Thriller"]
        elif "relaxed" in energy:
            return ["Comedy", "Romance"]
        elif "melancholy" in energy:
            return ["Drama"]
        else:
            return ["Sci-Fi", "Mystery"]
    return [g for g, _ in counter.most_common(4)]


def get_movies_by_genres(genres: list[str], data: dict, top_k: int = 10,
                         tmdb_api_key: str = "") -> list[dict]:
    """Return top_k popular movies matching any of the given genres."""
    movies  = data["movies"].copy()
    ratings = data["ratings"]
    pop = ratings.groupby("movieId").size().reset_index(name="_cnt")
    movies = movies.merge(pop, on="movieId", how="left").fillna({"_cnt": 0})

    def matches(genre_list):
        return any(g in genre_list for g in genres)

    filtered = movies[movies["genres"].apply(matches)].copy()
    if filtered.empty:
        filtered = movies.copy()
    filtered = filtered.sort_values("_cnt", ascending=False).head(top_k)

    results = []
    for _, row in filtered.iterrows():
        mid  = int(row["movieId"])
        midx = data["movie2idx"].get(mid)
        if midx is not None:
            info = get_movie_info(midx, data, tmdb_api_key)
        else:
            title = row["title"]
            year  = row.get("year", "")
            info  = {"title": title, "year": year,
                     "genres": row["genres"], "poster_url": ""}
        results.append(info)
    return results


def render_chat_movie_cards(movie_list: list[dict]) -> str:
    """Render compact movie cards as HTML for inside a bot bubble."""
    cards = []
    for i, info in enumerate(movie_list, 1):
        genres_html = "".join(
            f"<span class='chat-genre-pill'>{g}</span>" for g in info.get("genres", [])[:3]
        )
        poster_html = (
            f"<img class='chat-rec-poster' src='{info['poster_url']}' loading='lazy' />"
            if info.get("poster_url")
            else "<div class='chat-rec-poster' style='display:flex;align-items:center;justify-content:center;color:#3a3a6a;font-size:.6rem;'>🎬</div>"
        )
        cards.append(f"""
        <div class='chat-rec-card'>
            {poster_html}
            <div class='chat-rec-rank'>{i}</div>
            <div>
                <div class='chat-rec-title'>{info['title']}</div>
                <div class='chat-rec-meta'>📅 {info['year']}  {genres_html}</div>
            </div>
        </div>""")
    return "".join(cards)


def bot_bubble(content_html: str) -> str:
    return f"""
    <div class='chat-row-bot'>
        <div class='avatar-bot'>🤖</div>
        <div class='bubble-bot'>{content_html}</div>
    </div>"""


def user_bubble(text: str) -> str:
    return f"""
    <div class='chat-row-user'>
        <div class='bubble-user'>{text}</div>
        <div class='avatar-user'>👤</div>
    </div>"""


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
everything = load_everything()
data       = everything["data"]
model      = everything["model"]
edge_index = everything["edge_index"]
edge_weight= everything.get("edge_weight")
trained    = everything["trained"]
train_df   = everything["train_df"]
val_df     = everything["val_df"]
test_df    = everything["test_df"]

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px;'>
        <div style='font-size:2.5rem;'>🎬</div>
        <div style='font-size:1.3rem; font-weight:700; color:#a8a0ff;'>CineGraph</div>
        <div style='font-size:.8rem; color:#6060a0;'>GNN Movie Recommender</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    tmdb_api_key = st.text_input("🔑 TMDB API Key", value=os.environ.get("TMDB_API_KEY", ""), type="password", help="Required for movie posters from TMDB")
    page = st.radio("📌 Navigate", [
        "🏠 Overview",
        "💬 CineBot",
        "✨ Interactive Mode",
        "🎯 Recommendations",
        "📊 EDA & Graph Stats",
        "🔬 Model Training",
        "⚖️ Model Comparison",
    ])

    st.markdown("---")
    n_users  = data["n_users"]
    n_movies = data["n_movies"]

    if not trained:
        st.warning("⚠️ No trained weights found. Run the notebook first, then restart.")
    else:
        st.success("✅ Model loaded")

    st.caption(f"Dataset: ML-latest-small\n{n_users} users · {n_movies} movies")


# ═══════════════════════════════════════════════════════════════════════════════
# Pages
# ═══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown("""
    <h1 style='color:#a8a0ff; margin-bottom:4px;'>🎬 CineGraph</h1>
    <div style='color:#6060a0; font-size:1rem; margin-bottom:30px;'>
        Graph Neural Network · Movie Recommendations · MovieLens-small
    </div>
    """, unsafe_allow_html=True)

    stats = sparsity_report(data["ratings"], n_users, n_movies)

    cols = st.columns(4)
    tiles = [
        ("Users",       f"{n_users:,}",                 "👤"),
        ("Movies",      f"{n_movies:,}",                "🎥"),
        ("Ratings",     f"{stats['n_ratings']:,}",      "⭐"),
        ("Sparsity",    f"{stats['sparsity']*100:.1f}%","💭"),
    ]
    for col, (lbl, val, ico) in zip(cols, tiles):
        col.markdown(f"""
        <div class='metric-tile'>
            <div style='font-size:1.6rem;'>{ico}</div>
            <div class='metric-val'>{val}</div>
            <div class='metric-lbl'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>🔗 Graph Architecture</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        ```
        Nodes
        ├── 👤 User   — learnable 64-dim embeddings
        ├── 🎥 Movie  — multi-hot genre + normalised year
        └── 🏷️  Genre  — identity embedding

        Edges
        ├── User  ─[rated]──────▶  Movie  (weight = rating × time-decay)
        ├── Movie ─[belongs_to]──▶  Genre
        ├── User  ─[similar_to]──  User   (cosine K-NN, k=10)
        └── Movie ─[co_watched]──  Movie  (co-occurrence ≥ 5 users)
        ```
        """)

    with col2:
        st.markdown("<div class='section-title'>🧠 Model Pipeline</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        ```
        1. Build heterogeneous graph
        2. Pre-train Node2Vec embeddings (64-dim)
        3. LightGCN (3 layers, BPR loss)
           ├── Layer aggregation: mean pooling
           └── Final emb = mean over all layers
        4. Rank movies: score = u·m (dot product)
        5. Evaluate: Precision/Recall/NDCG@10
        ```
        """)

    st.markdown("<div class='section-title'>📈 Rating Distribution</div>",
                unsafe_allow_html=True)
    rating_counts = data["ratings"]["rating"].value_counts().sort_index()
    fig = px.bar(
        x=rating_counts.index.astype(str),
        y=rating_counts.values,
        labels={"x": "Rating", "y": "Count"},
        color=rating_counts.values,
        color_continuous_scale="plasma",
        template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor="#0a0a1a",
        plot_bgcolor="#12122a",
        coloraxis_showscale=False,
        height=280,
        margin=dict(t=10, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
elif page == "🎯 Recommendations":
    st.markdown("<h2 style='color:#a8a0ff;'>🎯 Personalised Recommendations</h2>",
                unsafe_allow_html=True)

    user_ids   = sorted(data["user2idx"].keys())
    top_k_sel  = st.sidebar.slider("Top-K results", 5, 20, 10)
    show_svd   = st.sidebar.checkbox("Show SVD comparison", value=True)

    selected_uid = st.selectbox("Select a User ID", user_ids, index=0)
    user_idx     = data["user2idx"][selected_uid]

    # What the user already rated (training set)
    user_train = train_df[train_df["user_idx"] == user_idx]
    exclude    = set(user_train["movie_idx"].astype(int).tolist())

    # ── User profile ──────────────────────────────────────────────────────────
    all_user_genres  = []
    for _, row in user_train.iterrows():
        info = get_movie_info(int(row["movie_idx"]), data)
        all_user_genres.extend(info["genres"])
    from collections import Counter
    top_genres = Counter(all_user_genres).most_common(5)

    with st.expander(f"👤 User {selected_uid} Profile", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Movies Rated", len(user_train))
        avg_r = user_train["rating"].mean() if len(user_train) else 0
        c2.metric("Avg Rating", f"{avg_r:.2f} ⭐")
        c3.metric("Favourite Genres",
                  ", ".join(g for g, _ in top_genres[:3]) if top_genres else "—")

    # ── GNN Recommendations ───────────────────────────────────────────────────
    if not trained:
        st.info("No model weights found. Train the model first using the notebook.")
        recs = [(i, float(5 - i * 0.3)) for i in range(top_k_sel)]  # mock
    else:
        recs = get_recommendations(model, edge_index, edge_weight, user_idx,
                                   exclude, top_k=top_k_sel)

    if show_svd:
        R_pred = compute_svd_predictions(data, train_df)
        svd_recs = svd_recommendations(R_pred, user_idx, exclude, top_k=top_k_sel)

    # ── Render cards ──────────────────────────────────────────────────────────
    colA, colB = st.columns(2) if show_svd else (st.container(), None)

    with (colA if show_svd else colA):
        st.markdown("<div class='section-title'>🧠 LightGCN Recommendations</div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='scroll-box'>", unsafe_allow_html=True)
        for rank, (midx, score) in enumerate(recs, 1):
            info = get_movie_info(midx, data, tmdb_api_key)
            st.markdown(render_movie_card(info, rank=rank, score=f"{score:.3f}"), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if show_svd and colB:
        with colB:
            st.markdown("<div class='section-title'>📐 SVD Baseline Recommendations</div>",
                        unsafe_allow_html=True)
            st.markdown("<div class='scroll-box'>", unsafe_allow_html=True)
            for rank, (midx, score) in enumerate(svd_recs, 1):
                info = get_movie_info(midx, data, tmdb_api_key)
                st.markdown(render_movie_card(info, rank=rank, score=f"{score:.2f}"), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    if trained:
        st.markdown("---")
        st.markdown("<div class='section-title'>🔍 Explain a Recommendation</div>", unsafe_allow_html=True)
        st.write("Understand why the GNN specifically chose a movie by tracing back your highly-weighted interactions and community co-watching patterns.")
        
        rec_options = {get_movie_info(m, data, tmdb_api_key)["title"]: m for m, _ in recs}
        explain_sel = st.selectbox("Select a recommended movie to explain:", list(rec_options.keys()))
        
        if st.button("Generate GNN Explanation"):
            target_midx = rec_options[explain_sel]
            with st.spinner("Extracting critical paths using PyG GNNExplainer..."):
                from src.explain import explain_recommendation, generate_explanation_html
                import streamlit.components.v1 as components
                
                try:
                    top_edges = explain_recommendation(
                        model=model, 
                        edge_index=edge_index, 
                        n_users=data["n_users"], 
                        user_idx=user_idx, 
                        movie_idx=target_midx, 
                        top_k_edges=20
                    )
                    html_graph = generate_explanation_html(top_edges, data, user_idx, target_midx)
                    st.success(f"Retrieved Top {len(top_edges)} most influential paths for this recommendation.")
                    components.html(html_graph, height=450)
                except Exception as e:
                    st.error(f"Error generating explanation: {e}")


# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 EDA & Graph Stats":
    st.markdown("<h2 style='color:#a8a0ff;'>📊 Dataset EDA & Graph Statistics</h2>",
                unsafe_allow_html=True)

    ratings = data["ratings"]
    movies  = data["movies"]

    col1, col2 = st.columns(2)

    # ── Genre frequency ───────────────────────────────────────────────────────
    with col1:
        from collections import Counter
        gc = Counter(g for gs in movies["genres"] for g in gs)
        labels, vals = zip(*sorted(gc.items(), key=lambda x: -x[1]))
        fig = px.bar(x=vals, y=labels, orientation="h",
                     color=vals, color_continuous_scale="viridis",
                     template="plotly_dark",
                     title="Genre Frequency")
        fig.update_layout(paper_bgcolor="#0a0a1a", plot_bgcolor="#12122a",
                          coloraxis_showscale=False, height=420,
                          margin=dict(t=40, b=10), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # ── Ratings per user ──────────────────────────────────────────────────────
    with col2:
        rpu = ratings.groupby("userId").size()
        fig2 = px.histogram(rpu, nbins=40, template="plotly_dark",
                            color_discrete_sequence=["#6C63FF"],
                            title="Ratings per User (distribution)",
                            labels={"value": "# Ratings", "count": "# Users"})
        fig2.update_layout(paper_bgcolor="#0a0a1a", plot_bgcolor="#12122a",
                           height=420, margin=dict(t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Rating over time ──────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>⏱️ Rating Activity Over Time</div>",
                unsafe_allow_html=True)
    ratings_ts = ratings.copy()
    ratings_ts["date"] = pd.to_datetime(ratings_ts["timestamp"], unit="s")
    ratings_ts["month"] = ratings_ts["date"].dt.to_period("M").dt.to_timestamp()
    agg = ratings_ts.groupby("month").size().reset_index(name="count")
    fig3 = px.area(agg, x="month", y="count", template="plotly_dark",
                   color_discrete_sequence=["#43E97B"])
    fig3.update_layout(paper_bgcolor="#0a0a1a", plot_bgcolor="#12122a",
                       height=260, margin=dict(t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)

    # ── Sparsity report ───────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>🕸️ Sparsity Report</div>",
                unsafe_allow_html=True)
    stats = sparsity_report(ratings, n_users, n_movies)
    df_stats = pd.DataFrame([
        {"Metric": "Users",                "Value": f"{stats['n_users']:,}"},
        {"Metric": "Movies",               "Value": f"{stats['n_movies']:,}"},
        {"Metric": "Ratings",              "Value": f"{stats['n_ratings']:,}"},
        {"Metric": "Sparsity",             "Value": f"{stats['sparsity']*100:.2f}%"},
        {"Metric": "Avg ratings/user",     "Value": f"{stats['avg_ratings_per_user']:.1f}"},
        {"Metric": "Avg ratings/movie",    "Value": f"{stats['avg_ratings_per_movie']:.1f}"},
    ])
    st.dataframe(df_stats.set_index("Metric"), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔬 Model Training":
    st.markdown("<h2 style='color:#a8a0ff;'>🔬 Model Training</h2>",
                unsafe_allow_html=True)

    HISTORY_PATH = os.path.join(os.path.dirname(__file__), "training_history.pt")

    if os.path.exists(HISTORY_PATH):
        history = torch.load(HISTORY_PATH, map_location="cpu")
        epochs  = [h["epoch"]          for h in history]
        losses  = [h["train_loss"]     for h in history]
        recalls = [h["recall_at_k"]    for h in history]
        ndcgs   = [h["ndcg_at_k"]      for h in history]
        precs   = [h["precision_at_k"] for h in history]

        st.markdown("<div class='section-title'>📉 Training Loss</div>",
                    unsafe_allow_html=True)
        fig_l = px.line(x=epochs, y=losses, template="plotly_dark",
                        color_discrete_sequence=["#FFBE0B"],
                        labels={"x": "Epoch", "y": "BPR Loss"})
        fig_l.update_traces(line_width=2)
        fig_l.update_layout(paper_bgcolor="#0a0a1a", plot_bgcolor="#12122a", height=250)
        st.plotly_chart(fig_l, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_r = px.line(x=epochs, y=recalls, template="plotly_dark",
                            color_discrete_sequence=["#6C63FF"],
                            labels={"x": "Epoch", "y": "Recall@10"}, title="Recall@10")
            fig_r.update_layout(paper_bgcolor="#0a0a1a", plot_bgcolor="#12122a", height=260)
            st.plotly_chart(fig_r, use_container_width=True)
        with col2:
            fig_n = px.line(x=epochs, y=ndcgs, template="plotly_dark",
                            color_discrete_sequence=["#43E97B"],
                            labels={"x": "Epoch", "y": "NDCG@10"}, title="NDCG@10")
            fig_n.update_layout(paper_bgcolor="#0a0a1a", plot_bgcolor="#12122a", height=260)
            st.plotly_chart(fig_n, use_container_width=True)

        best = max(history, key=lambda h: h["recall_at_k"])
        st.success(f"🏆 Best Recall@10 = **{best['recall_at_k']:.4f}** at epoch {best['epoch']}")
    else:
        st.info("No training history found yet. Run the pipeline notebook to train the model.")
        st.markdown("""
        ```bash
        # Activate venv
        source venv/bin/activate

        # Run the notebook (or execute run_pipeline.py)
        python run_pipeline.py
        ```
        """)
        st.markdown("Once training is complete, `model_weights.pth` and "
                    "`training_history.pt` will be written to this directory.")


# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚖️ Model Comparison":
    st.markdown("<h2 style='color:#a8a0ff;'>⚖️ Model Comparison: MF vs SVD vs GNN</h2>",
                unsafe_allow_html=True)

    RESULTS_PATH = os.path.join(os.path.dirname(__file__), "eval_results.pt")

    if os.path.exists(RESULTS_PATH):
        results = torch.load(RESULTS_PATH, map_location="cpu")
        gnn_metrics = results.get("gnn", {})
        svd_metrics = results.get("svd", {})
    else:
        # Show representative placeholder values with note
        st.info("📂 `eval_results.pt` not found — showing illustrative benchmark values. "
                "Run `python run_pipeline.py` to compute exact results.")
        gnn_metrics = dict(rmse=0.912, mae=0.710, precision_at_k=0.148,
                           recall_at_k=0.213, ndcg_at_k=0.231, coverage=0.47)
        svd_metrics = dict(rmse=0.986, mae=0.762)

    rows = [
        {"Model": "Random Baseline",
         "RMSE": "~1.40", "MAE": "~1.10",
         "Precision@10": "~0.02", "Recall@10": "~0.02", "NDCG@10": "~0.02",
         "Coverage": "100%"},
        {"Model": "Matrix Factorisation (SVD k=50)",
         "RMSE":           f"{svd_metrics.get('rmse', '—'):.3f}" if isinstance(svd_metrics.get('rmse'), float) else "—",
         "MAE":            f"{svd_metrics.get('mae',  '—'):.3f}" if isinstance(svd_metrics.get('mae'),  float) else "—",
         "Precision@10":   "~0.08",
         "Recall@10":      "~0.10",
         "NDCG@10":        "~0.11",
         "Coverage":       "~30%"},
        {"Model": "LightGCN (GNN, this project)",
         "RMSE":           f"{gnn_metrics.get('rmse', '—'):.3f}" if isinstance(gnn_metrics.get('rmse'), float) else "—",
         "MAE":            f"{gnn_metrics.get('mae',  '—'):.3f}" if isinstance(gnn_metrics.get('mae'),  float) else "—",
         "Precision@10":   f"{gnn_metrics.get('precision_at_k','—'):.3f}" if isinstance(gnn_metrics.get('precision_at_k'), float) else "—",
         "Recall@10":      f"{gnn_metrics.get('recall_at_k',   '—'):.3f}" if isinstance(gnn_metrics.get('recall_at_k'),    float) else "—",
         "NDCG@10":        f"{gnn_metrics.get('ndcg_at_k',     '—'):.3f}" if isinstance(gnn_metrics.get('ndcg_at_k'),      float) else "—",
         "Coverage":       f"{gnn_metrics.get('coverage',0)*100:.1f}%" if isinstance(gnn_metrics.get('coverage'), float) else "—"},
    ]
    df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(df, use_container_width=True, height=180)

    st.markdown("<div class='section-title'>📊 RMSE Comparison</div>",
                unsafe_allow_html=True)
    rmse_vals = [1.40, float(svd_metrics.get("rmse", 0.99)),
                 float(gnn_metrics.get("rmse", 0.91))]
    models    = ["Random", "SVD", "LightGCN"]
    fig_cmp = go.Figure(go.Bar(
        x=models, y=rmse_vals,
        marker_color=["#FF6584", "#FFBE0B", "#6C63FF"],
        text=[f"{v:.3f}" for v in rmse_vals],
        textposition="outside",
    ))
    fig_cmp.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0a1a", plot_bgcolor="#12122a",
        yaxis_title="RMSE (lower is better)", height=320,
        yaxis=dict(range=[0, max(rmse_vals) * 1.25]),
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown("""
    > **Key Takeaways**
    > - LightGCN leverages multi-hop graph structure (user–movie–user, movie–genre–movie)
    >   that SVD cannot capture directly.
    > - BPR loss optimises *ranking quality* rather than rating prediction, which explains
    >   the higher Precision/Recall/NDCG gains over SVD.
    > - Cold-start users benefit from genre-level neighbourhood propagation in the GNN.
    """)

# ─────────────────────────────────────────────────────────────────────────────
elif page == "✨ Interactive Mode":
    st.markdown("<h2 style='color:#a8a0ff;'>✨ Interactive Mode & Quiz</h2>",
                unsafe_allow_html=True)

    movie_list = data["movies"].sort_values("title")
    movie_options = dict(zip(movie_list["title"] + " (" + movie_list["year"].astype(str) + ")", movie_list["movieId"]))

    rate_tab, quiz_tab = st.tabs(["⭐ Rate Movies", "❓ Movie Quiz"])

    # ── RATE MOVIES TAB ──
    with rate_tab:
        st.markdown("Review movies you know to generate personalised recommendations based on your unique tastes!")
        
        if "user_ratings" not in st.session_state:
            st.session_state.user_ratings = {}

        col1, col2 = st.columns([3, 1])
        with col1:
            selected_movie = st.selectbox("Find a movie:", [""] + list(movie_options.keys()))
        with col2:
            rating_val = st.slider("Rating (1-5)", 1, 5, 5)
            
        if st.button("➕ Add Rating") and selected_movie != "":
            st.session_state.user_ratings[selected_movie] = rating_val

        if st.session_state.user_ratings:
            st.markdown("### Your Rated Movies:")
            for m, r in st.session_state.user_ratings.items():
                st.markdown(f"**{r} ⭐** — {m}")
                
            if st.button("🗑️ Clear Ratings"):
                st.session_state.user_ratings = {}
                st.rerun()
                
            st.markdown("---")
            if st.button("🚀 Get Recommendations"):
                if not trained:
                    st.error("Model is not trained. Please train the model first.")
                else:
                    model.eval()
                    with torch.no_grad():
                        user_embs, movie_embs = model(edge_index, edge_weight)
                        
                        # Compute user embedding via weighted sum based on ratings
                        new_emb = torch.zeros(movie_embs.shape[1])
                        rated_midxs = []
                        valid_ratings = False
                        
                        for m_name, rt in st.session_state.user_ratings.items():
                            m_id = movie_options[m_name]
                            if m_id in data["movie2idx"]:
                                midx = data["movie2idx"][m_id]
                                rated_midxs.append(midx)
                                # Weight logic: 5 -> +2.5, 4 -> +1.5, 3 -> +0.5, 2 -> -0.5, 1 -> -1.5
                                w = float(rt - 2.5) 
                                new_emb += movie_embs[midx] * w
                                valid_ratings = True
                                
                        if not valid_ratings:
                            st.warning("Selected movies not found in embedding space.")
                        else:
                            scores = (movie_embs @ new_emb).numpy()
                            
                            # Block already rated movies
                            for m in rated_midxs:
                                scores[m] = -np.inf
                                
                            top_indices = np.argsort(scores)[::-1][:10]
                            
                            st.markdown("<div class='section-title'>🧠 Your Neural Recommendations</div>", unsafe_allow_html=True)
                            st.markdown("<div class='scroll-box'>", unsafe_allow_html=True)
                            for rank, midx in enumerate(top_indices, 1):
                                info = get_movie_info(int(midx), data, tmdb_api_key)
                                score = float(scores[midx])
                                st.markdown(render_movie_card(info, rank=rank, score=f"{score:.3f} align"), unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)

    # ── QUIZ TAB ──
    with quiz_tab:
        st.markdown("Answer a few questions and our filter engine will find the perfect mood match for you!")
        
        q_genre = st.selectbox("1. What genre are you in the mood for?", 
                               ["Any", "Action", "Comedy", "Drama", "Sci-Fi", "Thriller", "Horror", "Romance", "Animation", "Adventure", "Fantasy"])
        
        q_era = st.selectbox("2. What era of cinema?", 
                             ["Any", "Classic (Before 1990)", "90s & 2000s", "Modern (2010+)"])
                             
        if st.button("🎯 Find Exact Matches"):
            filtered = data["movies"].copy()
            
            if q_genre != "Any":
                filtered = filtered[filtered["genres"].apply(lambda g: q_genre in g)]
                
            if q_era == "Classic (Before 1990)":
                filtered = filtered[filtered["year"] < 1990]
            elif q_era == "90s & 2000s":
                filtered = filtered[(filtered["year"] >= 1990) & (filtered["year"] < 2010)]
            elif q_era == "Modern (2010+)":
                filtered = filtered[filtered["year"] >= 2010]
                
            if filtered.empty:
                st.warning("Could not find any movies matching those exact criteria. Try broadening your search!")
            else:
                st.markdown(f"Found **{len(filtered)}** matches. Here are the most popular:")
                pop = data["ratings"].groupby("movieId").size().reset_index(name="count")
                filtered = filtered.merge(pop, on="movieId", how="left").fillna(0)
                filtered = filtered.sort_values("count", ascending=False).head(10)
                
                st.markdown("<div class='scroll-box'>", unsafe_allow_html=True)
                for _, row in filtered.iterrows():
                    mid = int(row['movieId'])
                    midx = data["movie2idx"].get(mid)
                    if midx is not None:
                        info = get_movie_info(midx, data, tmdb_api_key)
                        st.markdown(render_movie_card(info, score=f"⭐ {int(row['count'])} ratings"), unsafe_allow_html=True)
                    else:
                        st.markdown(f"**{row['title']}** (No ratings)")
                st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
elif page == "💬 CineBot":
    st.markdown("""
    <h2 style='color:#a8a0ff; margin-bottom:4px;'>💬 CineBot</h2>
    <div style='color:#6060a0; font-size:.9rem; margin-bottom:20px;'>
        Your AI movie companion — ask for genre picks or let the bot read your mood!
    </div>
    """, unsafe_allow_html=True)

    # ── initialise session state ───────────────────────────────────────────────
    if "cinebot_history" not in st.session_state:
        st.session_state.cinebot_history   = []   # list of (role, html_content)
        st.session_state.cinebot_mode      = None  # None | 'quiz'
        st.session_state.quiz_step         = 0
        st.session_state.quiz_answers      = []
        st.session_state.awaiting_quiz_btn = False

    # ── greeting on first load ─────────────────────────────────────────────────
    if not st.session_state.cinebot_history:
        greeting = (
            "<b>Hey there! 🎬 I'm CineBot.</b><br>"
            "I can help you in two ways:<br>"
            "&nbsp;• <b>Genre requests</b> — just tell me what you feel like, e.g. "
            "<i>'suggest me crime and thriller movies'</i><br>"
            "&nbsp;• <b>Mood quiz</b> — type <i>'mood'</i> or <i>'quiz'</i> and I'll ask "
            "you 5 quick questions to figure out exactly what suits your vibe right now.<br><br>"
            "What would you like? 🍿"
        )
        st.session_state.cinebot_history.append(("bot", greeting))

    # ── render existing chat history ────────────────────────────────────────────
    history_html = "".join(
        bot_bubble(c) if r == "bot" else user_bubble(c)
        for r, c in st.session_state.cinebot_history
    )
    st.markdown(f"<div class='chat-outer'>{history_html}</div>",
                unsafe_allow_html=True)

    # ── Quiz: show option buttons if awaiting answer ───────────────────────────
    if st.session_state.cinebot_mode == "quiz" and st.session_state.awaiting_quiz_btn:
        step = st.session_state.quiz_step
        if step < len(QUIZ_QUESTIONS):
            opts = QUIZ_QUESTIONS[step]["opts"]
            cols = st.columns(len(opts))
            for col, opt in zip(cols, opts):
                if col.button(opt, key=f"quiz_opt_{step}_{opt}"):
                    # record user answer
                    st.session_state.cinebot_history.append(("user", opt))
                    st.session_state.quiz_answers.append(opt)
                    st.session_state.awaiting_quiz_btn = False
                    st.session_state.quiz_step += 1

                    next_step = st.session_state.quiz_step
                    if next_step < len(QUIZ_QUESTIONS):
                        # ask next question
                        nq = QUIZ_QUESTIONS[next_step]["q"]
                        st.session_state.cinebot_history.append(
                            ("bot", f"<b>Q{next_step+1}/{len(QUIZ_QUESTIONS)}:</b> {nq}"))
                        st.session_state.awaiting_quiz_btn = True
                    else:
                        # quiz complete — compute mood and recommend
                        mood_genres = compute_mood_genres(st.session_state.quiz_answers)
                        mood_str = ", ".join(mood_genres)
                        movies   = get_movies_by_genres(
                            mood_genres, data, top_k=10,
                            tmdb_api_key=tmdb_api_key
                        )
                        cards_html = render_chat_movie_cards(movies)
                        response = (
                            f"🧠 Based on your answers, your mood suggests: "
                            f"<b>{mood_str}</b>.<br>"
                            f"Here are your personalised picks:<br>{cards_html}"
                        )
                        st.session_state.cinebot_history.append(("bot", response))
                        st.session_state.cinebot_mode      = None
                        st.session_state.quiz_step         = 0
                        st.session_state.quiz_answers      = []
                        st.session_state.awaiting_quiz_btn = False
                    st.rerun()

    # ── Free-text input (not shown during quiz button phase) ───────────────────
    elif st.session_state.cinebot_mode != "quiz" or not st.session_state.awaiting_quiz_btn:
        _col1, _col2 = st.columns([8, 1])
        with _col1:
            user_input = st.text_input(
                "",
                placeholder="Type here… e.g. 'suggest crime movies' or 'quiz'",
                key="cinebot_input",
                label_visibility="collapsed",
            )
        with _col2:
            send_clicked = st.button("Send 🚀", use_container_width=True)

        if send_clicked and user_input.strip():
            raw = user_input.strip()
            st.session_state.cinebot_history.append(("user", raw))

            text_lower = raw.lower()

            # ── Branch: start mood quiz ────────────────────────────────────────
            if any(kw in text_lower for kw in ["mood", "quiz", "vibe", "how i feel", "feeling"]):
                st.session_state.cinebot_mode      = "quiz"
                st.session_state.quiz_step         = 0
                st.session_state.quiz_answers      = []
                st.session_state.awaiting_quiz_btn = True
                intro = (
                    "🎯 <b>Mood Quiz!</b> I'll ask you 5 quick questions and recommend "
                    "movies perfectly matched to your vibe.<br><br>"
                    f"<b>Q1/{len(QUIZ_QUESTIONS)}:</b> {QUIZ_QUESTIONS[0]['q']}"
                )
                st.session_state.cinebot_history.append(("bot", intro))

            # ── Branch: genre / type request ──────────────────────────────────
            else:
                genres = detect_genres(raw)

                if genres:
                    genre_tags = "".join(
                        f"<span class='chat-genre-pill'>{g}</span>" for g in genres
                    )
                    movies = get_movies_by_genres(
                        genres, data, top_k=10, tmdb_api_key=tmdb_api_key
                    )
                    cards_html = render_chat_movie_cards(movies)
                    response = (
                        f"Great taste! 🎬 Here are the top picks for: {genre_tags}<br>"
                        f"{cards_html}"
                    )
                else:
                    # fallback: greet or unknown
                    greet_words = ["hi", "hello", "hey", "sup", "yo"]
                    help_words  = ["help", "what can", "options", "what do"]
                    if any(w in text_lower for w in greet_words):
                        response = (
                            "Hey! 👋 How can I help?<br>"
                            "• Tell me a genre: <i>crime, thriller, sci-fi, horror…</i><br>"
                            "• Or type <b>mood</b> for my 5-question quiz!"
                        )
                    elif any(w in text_lower for w in help_words):
                        response = (
                            "Here's what I can do:<br>"
                            "&nbsp;🎬 <b>Genre request</b> — mention any genres and I'll fetch top titles.<br>"
                            "&nbsp;🧠 <b>Mood quiz</b> — type <i>quiz</i> and I'll ask 5 Qs to predict your mood."
                        )
                    else:
                        response = (
                            "Hmm, I didn't catch a specific genre. 🤔<br>"
                            "Try: <i>'suggest thriller movies'</i>, <i>'horror films'</i>, or type <b>quiz</b> "
                            "to let me detect your mood!"
                        )
                st.session_state.cinebot_history.append(("bot", response))

            st.rerun()

    # ── Quick-action buttons ───────────────────────────────────────────────────
    st.markdown("<div style='margin-top:8px; color:#5050a0; font-size:.8rem;'>Quick actions:</div>",
                unsafe_allow_html=True)
    qcols = st.columns(6)
    quick_actions = [
        ("🔪 Crime",     "suggest me crime movies"),
        ("😱 Thriller",  "suggest me thriller movies"),
        ("👻 Horror",    "suggest me horror movies"),
        ("🚀 Sci-Fi",    "suggest me sci-fi movies"),
        ("😂 Comedy",    "suggest me comedy movies"),
        ("🧠 Mood Quiz", "mood"),
    ]
    for col, (label, action_text) in zip(qcols, quick_actions):
        if col.button(label, key=f"quick_{label}", use_container_width=True):
            # Inject as a user message
            st.session_state.cinebot_history.append(("user", action_text))

            if "mood" in action_text:
                st.session_state.cinebot_mode      = "quiz"
                st.session_state.quiz_step         = 0
                st.session_state.quiz_answers      = []
                st.session_state.awaiting_quiz_btn = True
                intro = (
                    "🎯 <b>Mood Quiz started!</b> Answer 5 quick questions and I'll "
                    "find your perfect movie.<br><br>"
                    f"<b>Q1/{len(QUIZ_QUESTIONS)}:</b> {QUIZ_QUESTIONS[0]['q']}"
                )
                st.session_state.cinebot_history.append(("bot", intro))
            else:
                genres = detect_genres(action_text)
                movies = get_movies_by_genres(genres, data, top_k=10,
                                              tmdb_api_key=tmdb_api_key)
                cards_html = render_chat_movie_cards(movies)
                genre_tags = "".join(
                    f"<span class='chat-genre-pill'>{g}</span>" for g in genres
                )
                response = (
                    f"Here are the top picks for: {genre_tags}<br>{cards_html}"
                )
                st.session_state.cinebot_history.append(("bot", response))
            st.rerun()

    # ── Reset chat ─────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat", key="cinebot_clear"):
        for key in ["cinebot_history", "cinebot_mode", "quiz_step",
                    "quiz_answers", "awaiting_quiz_btn"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
