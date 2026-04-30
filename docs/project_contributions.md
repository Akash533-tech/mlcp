# MLCP Project: Individual Contribution Breakdown

Here is a professional breakdown of the project divided into four balanced, highly technical roles. You can assign one of these roles to each team member for your report or presentation.

---

### Member 1: Data Engineering & Semantic Processing Lead
**Focus:** Data Pipeline, API Integration, and NLP.
*   **Data Ingestion & Cleaning:** Engineered the data pipeline to ingest the raw MovieLens dataset, resolving data sparsity and handling missing values across thousands of user interactions.
*   **Semantic Tag Processing (NLP):** Designed the `TagProcessor` using NLTK to evaluate user-generated text tags. Applied TF-IDF algorithms and VADER sentiment analysis to extract the underlying semantic weight of movie tags.
*   **External API Integration:** Integrated the external TMDB REST API to dynamically fetch real-time metadata and high-quality movie posters, ensuring the system remains lightweight and visually engaging.
*   **Graph Construction:** Developed the logic to convert flat relational CSV tables into a heavily connected, weighted bipartite graph (nodes and edges) ready for PyTorch Geometric processing.

### Member 2: Core Machine Learning & GNN Architect
**Focus:** PyTorch, LightGCN, and Model Optimization.
*   **GNN Architecture:** Designed and programmed the core `LightGCN` machine learning model using PyTorch Geometric, implementing multi-hop message passing to solve the collaborative filtering problem.
*   **Loss Function Engineering:** Implemented the Bayesian Personalized Ranking (BPR) optimization algorithm. Configured the negative sampling logic to train the model to effectively rank positive user interactions over unobserved ones.
*   **Hyperparameter Tuning:** Managed the deep learning training loop, optimizing learning rates, embedding dimensionalities (e.g., 64 dimensions), and weight decay to prevent model overfitting.
*   **Cold Start Calibration Logic:** Engineered the mathematical backend for the "Interactive Mode," utilizing Cosine Similarity and PyTorch tensor operations to instantly project new users into the latent space without retraining the model.

### Member 3: Baseline Modeling, Evaluation & Analytics Lead
**Focus:** Matrix Factorization, EDA, and System Metrics.
*   **Baseline SVD Model:** Developed a traditional Matrix Factorization model (Truncated SVD) using Scikit-Learn to serve as a rigorous empirical baseline against the advanced GNN model.
*   **Evaluation Metrics:** Designed the mathematical evaluation pipeline, calculating complex ranking metrics including **Recall@K** and **NDCG@K** to mathematically prove the performance uplift of the Graph Neural Network.
*   **Exploratory Data Analysis (EDA):** Conducted deep statistical analysis on the dataset. Identified the long-tail popularity bias in movie ratings and visualized the structural sparsity of the user-item interaction matrix.
*   **Psychographic Tuning Logic:** Implemented the backend mathematical logic to adjust recommendation scores dynamically based on popularity biases and the "Familiarity vs. Discovery" slider.

### Member 4: Frontend Architecture & UI/UX Developer
**Focus:** Streamlit, System Integration, and Explainability Visualization.
*   **Interactive Dashboard Design:** Built the entire interactive frontend architecture using Python's Streamlit framework, seamlessly bridging the gap between the complex PyTorch backend and the end user.
*   **Premium Aesthetic Overhaul:** Wrote custom HTML and CSS injections to bypass default styling, building a professional, high-end "Dark Mode" UI featuring dynamic layout cards and responsive movie grids.
*   **Neural Explanation Path Visualization:** Developed the "Explainability" module, creating interactive Node-Edge pathways (using PyVis and custom HTML) to visually demystify the GNN's "black box" and show exactly why a movie was recommended.
*   **State Management:** Engineered the complex `st.session_state` logic for the "Swipe Match" interface, handling real-time queues, user interactions, and immediate state rerendering without system lag.
