import requests
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from config import OMDB_API_KEY, RAWG_API_KEY

model = SentenceTransformer('all-MiniLM-L6-v2')

def fetch_movie_description(title):
    url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'Plot' in data and data['Plot'] != "N/A":
            return f"{data['Title']}: {data['Plot']} ({data.get('Genre', '')})"
    return title  # fallback if not found

def fetch_games(limit=10):
    url = f"https://api.rawg.io/api/games?key={RAWG_API_KEY}&page_size={limit}"
    response = requests.get(url)
    games = []
    if response.status_code == 200:
        for g in response.json().get('results', []):
            name = g['name']
            desc = g.get('slug', 'No description available.')  # RAWG often lacks full description
            games.append({'title': name, 'description': desc})
    return pd.DataFrame(games)

def get_game_suggestions(movie_input, game_df=None, top_n=3):
    if game_df is None:
        game_df = fetch_games(15)
    movie_description = fetch_movie_description(movie_input)
    movie_vec = model.encode([movie_description], convert_to_numpy=True)

    game_descriptions = game_df['description'].tolist()
    game_embeddings = model.encode(game_descriptions, convert_to_numpy=True)

    sims = cosine_similarity(movie_vec, game_embeddings)[0]
    top_indices = sims.argsort()[-top_n:][::-1]
    return [(game_df.iloc[i]['title'], sims[i]) for i in top_indices]

def save_feedback(movie, suggestion, feedback):
    data = {
        "movie": movie,
        "suggestion": suggestion,
        "feedback": feedback
    }
    try:
        with open('data/feedback.json', 'r') as f:
            existing = json.load(f)
    except FileNotFoundError:
        existing = []

    existing.append(data)
    with open('data/feedback.json', 'w') as f:
        json.dump(existing, f, indent=4)