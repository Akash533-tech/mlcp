# MLCP Project Defense & Tech Stack Guide

This guide is designed to help you prepare for your project defense. It breaks down the entire technology stack from the basics and provides a list of rigorous technical questions you should be prepared to answer.

---

## 1. The Technology Stack Explained

Here is the complete tech stack broken down simply, explaining **what** it is and **why** it was used in this specific project.

### Core Language & Data Handling
*   **Python**: The foundational programming language for the entire project. Chosen for its dominant ecosystem in machine learning and data science.
*   **Pandas & NumPy**: 
    *   *What it is*: Libraries for high-performance data manipulation and mathematical operations.
    *   *Why we used it*: To load, clean, and process the MovieLens dataset. Pandas handles the tabular data (merging `movies.csv`, `ratings.csv`, and `tags.csv`), while NumPy handles the underlying matrix and array computations efficiently.

### Machine Learning & Graph Neural Networks
*   **PyTorch**:
    *   *What it is*: A powerful, open-source deep learning framework developed by Meta. It allows for dynamic computation graphs (calculating math operations on the fly).
    *   *Why we used it*: To define, train, and optimize our machine learning models (the embeddings and the neural network layers).
*   **PyTorch Geometric (PyG)**:
    *   *What it is*: An extension library for PyTorch specifically designed for **Deep Learning on Graphs**.
    *   *Why we used it*: Traditional neural networks expect flat data (like images or text). Our data is a **Graph** (Users connected to Movies via Ratings). PyG provides the specific algorithms (like `LightGCN`) that allow neural networks to "pass messages" along these connections to learn how similar a user is to a movie.
*   **Scikit-Learn**:
    *   *What it is*: A standard machine learning library in Python.
    *   *Why we used it*: We used its implementation of Truncated SVD (Singular Value Decomposition) as a **baseline model** to compare our advanced GNN against. It proves *why* the GNN is better.

### Natural Language Processing (NLP)
*   **NLTK (VADER Sentiment)**:
    *   *What it is*: Natural Language Toolkit, a library for processing human language. VADER is a specific tool within it that scores text based on sentiment (positive/negative/neutral).
    *   *Why we used it*: To analyze the textual "tags" that users leave on movies. Instead of just knowing a user watched a movie, we extract the *semantic meaning* of their tags to give the graph more context (e.g., adding "dark" or "atmospheric" to the movie's profile).

### Frontend & Visualization
*   **Streamlit**:
    *   *What it is*: A framework that turns Python scripts into interactive web applications instantly.
    *   *Why we used it*: To build the entire interactive dashboard (Overview, Recommendations, Training, Interactive Mode) without needing to write a separate React/Node.js backend. It connects directly to our PyTorch model in real-time.
*   **Plotly**:
    *   *What it is*: An interactive graphing library.
    *   *Why we used it*: To plot the model training progress (BPR Loss, Recall, NDCG) so the user can visually see the model learning over time.

---

## 2. Professor Profile & Expected Questions

**Analysis of the Professor (Dr. Satish Shankarrao Panait):**
Based on his formal attire (suit, crisp collar), glasses, and composed, serious posture, he presents the image of a traditional, rigorous academic. Professors with this demeanor typically do not care for buzzwords (like just saying "AI" or "Deep Learning"). They want to know that you understand the **underlying mathematics, the fundamental computer science concepts, and the "why" behind your engineering choices.** 

If he is strict and technically focused, he will likely drill down into the mechanics of the algorithms rather than just looking at the pretty UI. 

### High-Probability Technical Questions to Prepare For:

#### A. Graph Neural Networks (GNN) vs Traditional ML
1.  **"Why did you use a Graph Neural Network instead of standard Collaborative Filtering or Matrix Factorization?"**
    *   *How to answer*: Explain that traditional Matrix Factorization (like SVD) only looks at direct, first-degree interactions (User A rated Movie B). A GNN captures *high-order connectivity* (User A rated Movie B, which was rated by User C, who rated Movie D). This neighborhood aggregation helps solve the cold-start problem much better.
2.  **"Explain the 'Message Passing' mechanism in your model."**
    *   *How to answer*: Explain that in PyTorch Geometric, nodes (users/movies) update their embeddings by gathering and combining the embeddings of their direct neighbors. 
3.  **"What is LightGCN, and why use it over a standard GCN?"**
    *   *How to answer*: Standard GCNs use non-linear activation functions (like ReLU) and feature transformations at every layer. LightGCN removes these, proving that for recommendation systems, simply smoothing the embeddings linearly across the graph is more efficient and prevents over-smoothing.

#### B. Loss Functions & Optimization
4.  **"I see you used BPR (Bayesian Personalized Ranking) Loss. Explain the mathematical intuition behind it. Why didn't you use Mean Squared Error (MSE)?"**
    *   *How to answer*: MSE is for *rating prediction* (guessing if a user gives a 4 or 5). BPR is for *ranking*. In recommendations, we don't care about the exact rating number; we just care that the model ranks a movie the user *will* like higher than a movie they *won't* like. BPR maximizes the margin between a positive sample and a negative sample.
5.  **"How did you handle negative sampling in this graph?"**
    *   *How to answer*: For every positive edge (a movie a user watched), we randomly sample a movie the user has *not* watched to act as the negative edge, training the model to push their embeddings apart.

#### C. Metrics and Evaluation
6.  **"You are using NDCG and Recall@K to evaluate your model. Define NDCG and explain why it is a better metric than just plain Accuracy."**
    *   *How to answer*: Accuracy is useless in recommendations because the data is highly imbalanced (a user hasn't seen 99% of movies). NDCG (Normalized Discounted Cumulative Gain) measures *ranking quality*. It gives a higher score if the relevant movies appear at the very top of the list (e.g., position 1 or 2) rather than further down (position 9 or 10).

#### D. System Design & Cold Start
7.  **"How does your 'Interactive Mode' actually work under the hood? When I rate a movie on the UI, what happens to the math?"**
    *   *How to answer*: When a user rates new movies, we don't retrain the entire model. Instead, we take the pre-trained embeddings of the movies they selected, multiply them by the rating weight, and aggregate (average) them to instantaneously project a new "User Vector" into the latent space. We then compute the dot product between this new vector and all other movie vectors to find the closest matches.
