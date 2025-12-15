from fastapi import FastAPI
from app.recommender import SongRecommender

app = FastAPI(title="Song Recommendation API")

recommender = SongRecommender("data/spotify_millsongdata.csv")

@app.get("/")
def home():
    return {"message": "Song Recommendation System API"}


@app.get("/recommend")
def recommend(song: str):
    results = recommender.recommend(song)
    if not results:
        return {"error": "Song not found"}
    return {
        "input_song": song,
        "recommended_songs": results
    }
