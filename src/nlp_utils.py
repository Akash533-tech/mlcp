import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


class TagProcessor:
    """
    Handles unstructured tag data by cleaning, deduplicating,
    scoring sentiment (VADER), and extracting semantic relevance (TF-IDF).
    """
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def clean_tags(self, tags_df: pd.DataFrame) -> pd.DataFrame:
        """Lowercase, strip, and remove empty tags."""
        tags = tags_df.copy()
        tags['tag'] = tags['tag'].astype(str).str.lower().str.strip()
        tags = tags[tags['tag'] != '']
        tags = tags[tags['tag'] != 'nan']
        return tags

    def compute_tag_weights(self, tags_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the TF-IDF weight and VADER sentiment for each Movie-Tag pair.
        Returns a DataFrame grouped by (movieId, tag) with a final 'weight' column.
        
        Final Weight = TF-IDF * (1 + compound_sentiment)
        """
        if tags_df.empty:
            return pd.DataFrame(columns=['movieId', 'tag', 'weight', 'sentiment', 'tfidf'])
            
        # Group by Movie and Tag to get term frequency count
        tf = tags_df.groupby(['movieId', 'tag']).size().reset_index(name='count')
        
        # Document Frequency (movies containing tag t)
        total_movies = tags_df['movieId'].nunique()
        doc_freq = tags_df.groupby('tag')['movieId'].nunique().reset_index(name='df')
        
        # IDF = log(N / df)
        doc_freq['idf'] = np.log1p(total_movies / doc_freq['df'])
        
        # Merge TF and IDF
        weights = tf.merge(doc_freq, on='tag')
        
        # Normalize TF by total tags for that movie
        total_tags_per_movie = tags_df.groupby('movieId').size().reset_index(name='total_tags')
        weights = weights.merge(total_tags_per_movie, on='movieId')
        weights['tf_norm'] = weights['count'] / weights['total_tags']
        weights['tfidf'] = weights['tf_norm'] * weights['idf']
        
        # Map Sentiment
        unique_tags = weights['tag'].unique()
        sentiments = {t: self.sia.polarity_scores(t)['compound'] for t in unique_tags}
        weights['sentiment'] = weights['tag'].map(sentiments)
        
        # Compute final edge weight
        # Ensure sentiment amplifier keeps weight strictly positive 
        # Range of sentiment is [-1, 1], so (1 + sentiment) is [0, 2]
        # We clip lower to 0.1 so even extremely negative tags have a structural edge, just weak.
        weights['weight'] = weights['tfidf'] * (1.0 + weights['sentiment']).clip(lower=0.1)
        
        return weights[['movieId', 'tag', 'weight', 'sentiment', 'tfidf']]
