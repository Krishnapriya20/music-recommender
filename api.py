from fastapi import FastAPI
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = FastAPI()

# -----------------------------
# LOAD MODEL
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("songs_clustered.csv")
features = df[["energy", "valence", "tempo"]]

scaler = StandardScaler()
X = scaler.fit_transform(features)

# -----------------------------
# HELPER: ALBUM COVER
# -----------------------------
def get_album(song, artist):
    try:
        query = f"{song} {artist}".replace(" ", "+")
        url = f"https://itunes.apple.com/search?term={query}&limit=1"
        res = requests.get(url).json()

        if res["resultCount"] > 0:
            return res["results"][0].get("artworkUrl100")
    except:
        pass

    return None

# -----------------------------
# HELPER: MOOD VECTOR
# -----------------------------
def get_mood_vector(text):

    moods = {
        "happy": "upbeat energetic fun music",
        "sad": "slow emotional sad music",
        "chill": "calm relaxing soft music",
        "energetic": "high energy workout hype music"
    }

    user_emb = model.encode([text])[0]

    best_match = max(
        moods,
        key=lambda m: cosine_similarity(
            [user_emb],
            [model.encode([moods[m]])[0]]
        )[0][0]
    )

    mapping = {
        "happy": [0.9, 0.9, 120],
        "sad": [0.2, 0.3, 60],
        "chill": [0.4, 0.5, 70],
        "energetic": [0.9, 0.6, 150]
    }

    return mapping[best_match]

# -----------------------------
# MAIN ENDPOINT
# -----------------------------
@app.get("/recommend")
def recommend(mood: str):

    # 1. Convert mood → vector
    mood_values = get_mood_vector(mood)

    mood_vector = scaler.transform(
        pd.DataFrame([mood_values], columns=features.columns)
    )

    # 2. Compute similarity
    df["score"] = cosine_similarity(mood_vector, X)[0]

    # 3. Get top songs
    results = df.sort_values("score", ascending=False).head(10)

    # 4. Format response
    output = []

    for _, row in results.iterrows():
        song = row["track_name"]
        artist = row["artist_name"]

        output.append({
            "song": song,
            "artist": artist,
            "image": get_album(song, artist),
            "youtube": f"https://www.youtube.com/results?search_query={song.replace(' ', '+')}"
        })

    return output