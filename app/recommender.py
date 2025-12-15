import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SongRecommender:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df = self.df.dropna(subset=["text"])

        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["text"])

    def recommend(self, song_title, top_n=10):
        matches = self.df[self.df["song"].str.lower() == song_title.lower()]

        if matches.empty:
            return {"error": f"Song '{song_title}' not found in dataset"}

        idx = matches.index[0]

        similarity_scores = cosine_similarity(
            self.tfidf_matrix[idx], self.tfidf_matrix
        ).flatten()

        similar_indices = similarity_scores.argsort()[::-1][1:top_n+1]

        return self.df.iloc[similar_indices][["song", "artist"]].to_dict(orient="records")
