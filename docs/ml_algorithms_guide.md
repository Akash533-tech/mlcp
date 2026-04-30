# MLCP Algorithms Breakdown

Here is a comprehensive, academically rigorous breakdown of all the Machine Learning and Data Science algorithms used in the MLCP project. This explanation is structured to focus on the mathematical intuition and the specific purpose of each algorithm in the pipeline.

---

### 1. LightGCN (Light Graph Convolutional Network)
**Category:** Deep Learning / Graph Neural Networks
*   **What it is:** LightGCN is a state-of-the-art model designed specifically for collaborative filtering. Unlike standard GCNs (which use feature transformations and non-linear activation functions like ReLU at every layer), LightGCN strips these out. It relies entirely on **linear neighborhood aggregation**.
*   **How it works:** It treats users and movies as nodes in a bipartite graph. The embedding of a user is updated by calculating the weighted sum of the embeddings of the movies they have interacted with. This "message passing" is repeated across multiple layers (e.g., 3 hops), allowing the model to capture high-order connectivity (User A -> Movie 1 -> User B -> Movie 2).
*   **Purpose in Project:** This is the core recommendation engine. It learns the latent representations (embeddings) of users and movies by understanding the structural topology of their interactions, which solves the sparsity and cold-start problems much better than traditional matrix factorization.

### 2. BPR (Bayesian Personalized Ranking)
**Category:** Optimization Algorithm / Loss Function
*   **What it is:** BPR is a pairwise ranking loss function derived from maximum posterior estimator mathematics. 
*   **How it works:** Instead of trying to predict the exact rating a user would give a movie (Pointwise prediction, like MSE), BPR takes a *positive item* (a movie the user watched) and a randomly sampled *negative item* (a movie they haven't watched). The algorithm updates the model weights to maximize the margin between the positive and negative items.
*   **Purpose in Project:** It optimizes the LightGCN model. Since recommendation systems are inherently about generating a ranked Top-K list rather than predicting numerical scores, BPR trains the model specifically to rank relevant movies higher than irrelevant ones.

### 3. Truncated SVD (Singular Value Decomposition)
**Category:** Matrix Factorization / Dimensionality Reduction
*   **What it is:** A classic linear algebra algorithm that reduces a large, sparse user-item interaction matrix ($R$) into lower-dimensional matrices ($U$ and $V$).
*   **How it works:** It approximates the original matrix by keeping only the top $k$ singular values, effectively compressing the data into a latent feature space. The dot product of a user's vector and a movie's vector in this reduced space gives the predicted interaction score.
*   **Purpose in Project:** This is used strictly as the **Baseline Comparison Model**. We compute recommendations using SVD to empirically demonstrate and measure how much better the modern Graph Neural Network (LightGCN) performs on metrics like NDCG and Recall.

### 4. TF-IDF (Term Frequency-Inverse Document Frequency)
**Category:** Natural Language Processing (NLP) / Statistical Measure
*   **What it is:** An algorithm that evaluates how important a specific word (or "tag") is to a document (or "movie") relative to a whole corpus.
*   **How it works:** It multiplies two metrics: 
    *   *Term Frequency (TF)*: How often a tag appears for a specific movie.
    *   *Inverse Document Frequency (IDF)*: How rare the tag is across *all* movies. (A tag like "movie" has low IDF, while "neo-noir" has high IDF).
*   **Purpose in Project:** Used in the `TagProcessor` to convert unstructured, user-generated text tags into weighted graph edges. It ensures that unique, highly descriptive tags create stronger connections in the graph than generic, heavily-used tags.

### 5. VADER Sentiment Analysis (Valence Aware Dictionary and sEntiment Reasoner)
**Category:** Natural Language Processing (NLP) / Rule-based Algorithm
*   **What it is:** A lexicon and rule-based sentiment analysis algorithm specifically tuned for social media and short texts.
*   **How it works:** It maps words to pre-computed sentiment scores (Valence) and applies grammatical and syntactical rules (like handling the word "not" or exclamation marks) to compute a normalized `compound` score between -1 (negative) and +1 (positive).
*   **Purpose in Project:** It acts as a multiplier for the TF-IDF weights. If a user tags a movie with "visually stunning" (Positive VADER score), the graph edge weight is strengthened. If a tag is heavily negative, the structural weight is penalized, giving the GNN deep semantic context rather than just treating all tags equally.
