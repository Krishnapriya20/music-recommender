import streamlit as st
import pandas as pd
import json
import os
import requests
import uuid
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Mood Console", layout="wide")

# -----------------------------
# 🎨 CUSTOM UI (CORE UPGRADE)
# -----------------------------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #0f0f1a, #1a1a2e);
    color: #eaeaea;
}

.stApp {
    background: transparent;
}

h1, h2, h3 {
    color: #ffffff;
    font-weight: 600;
}

section[data-testid="stSidebar"] {
    background: #0d0d18;
    border-right: 1px solid rgba(255,255,255,0.05);
}

.card {
    background: rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 16px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    transition: all 0.25s ease;
}

.card:hover {
    transform: translateY(-4px);
    border: 1px solid rgba(255,255,255,0.2);
}

.song-title {
    font-size: 18px;
    font-weight: 600;
}

.artist {
    font-size: 13px;
    color: #bbbbbb;
}

.link-btn {
    font-size: 13px;
    color: #8ab4ff;
    text-decoration: none;
}

.link-btn:hover {
    text-decoration: underline;
}

.stat-box {
    background: rgba(255,255,255,0.05);
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -----------------------------
# FILES
# -----------------------------
USER_FILE = "data_store/users.json"
STATS_FILE = "data_store/stats.json"

def load_json(file, default):
    if not os.path.exists(file):
        return default
    with open(file, "r") as f:
        try:
            return json.load(f)
        except:
            return default

def save_json(file, data):
    with open(file, "w") as f:
        json.dump(data, f, indent=4)

users = load_json(USER_FILE, {})
stats = load_json(STATS_FILE, {"total_users": 0, "total_actions": 0})

# -----------------------------
# TRACKING
# -----------------------------
def log_action(user, action, song):
    entry = {
        "user": user,
        "action": action,
        "song": song,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    stats["total_actions"] += 1
    users[user]["actions"].append(entry)

    save_json(USER_FILE, users)
    save_json(STATS_FILE, stats)

# -----------------------------
# ALBUM
# -----------------------------
@st.cache_data
def get_album_cover(song, artist):
    try:
        query = f"{song} {artist}".replace(" ", "+")
        url = f"https://itunes.apple.com/search?term={query}&limit=5"
        res = requests.get(url).json()

        for item in res.get("results", []):
            if song.lower() in item.get("trackName", "").lower():
                return item.get("artworkUrl100")

        return res["results"][0].get("artworkUrl100")
    except:
        return None

# -----------------------------
# MOOD
# -----------------------------
def get_mood_vector(text):
    moods = {
        "happy": "upbeat energetic fun music",
        "sad": "slow emotional sad music",
        "chill": "calm relaxing soft music",
        "energetic": "high energy workout hype music"
    }

    user_emb = model.encode([text])[0]

    best = max(moods, key=lambda m:
        cosine_similarity([user_emb], [model.encode([moods[m]])[0]])[0][0]
    )

    return {
        "happy": [0.9,0.9,120],
        "sad": [0.2,0.3,60],
        "chill": [0.4,0.5,70],
        "energetic": [0.9,0.6,150]
    }[best]

# -----------------------------
# DATA
# -----------------------------
df = pd.read_csv("songs_clustered.csv")
features = df[["energy","valence","tempo"]]
scaler = StandardScaler()
X = scaler.fit_transform(features)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🎛 Mood Console")

user_id = st.sidebar.text_input("Identity")
mood_input = st.sidebar.text_input("Describe feeling")

if user_id and user_id not in users:
    users[user_id] = {
        "liked": [],
        "actions": [],
        "share_token": str(uuid.uuid4())[:6],
        "created_at": str(datetime.now())
    }
    stats["total_users"] += 1
    save_json(USER_FILE, users)
    save_json(STATS_FILE, stats)

if user_id:
    token = users[user_id]["share_token"]
    st.sidebar.code(f"http://localhost:8501/?share={token}")

st.sidebar.markdown("### System Stats")
st.sidebar.markdown(f"<div class='stat-box'>👥 {stats['total_users']} Users</div>", unsafe_allow_html=True)
st.sidebar.markdown(f"<div class='stat-box'>⚡ {stats['total_actions']} Actions</div>", unsafe_allow_html=True)

run = st.sidebar.button("Analyze Mood")

# -----------------------------
# HEADER
# -----------------------------
st.title("🎧 Mood Console")
st.caption("A system that adapts to emotion, not clicks.")

# -----------------------------
# RECOMMEND
# -----------------------------
if run and user_id and mood_input:

    mood = get_mood_vector(mood_input)
    mood_vec = scaler.transform(pd.DataFrame([mood], columns=features.columns))

    df["score"] = cosine_similarity(mood_vec, X)[0]
    results = df.sort_values("score", ascending=False).head(10)

    for i, row in results.iterrows():
        song = row["track_name"]
        artist = row["artist_name"]

        img = get_album_cover(song, artist)
        yt = f"https://youtube.com/results?search_query={song.replace(' ','+')}"

        st.markdown(f"<div class='card'>", unsafe_allow_html=True)

        col1, col2 = st.columns([1,3])

        with col1:
            if img:
                st.image(img, width=90)

        with col2:
            st.markdown(f"<div class='song-title'>{song}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='artist'>{artist}</div>", unsafe_allow_html=True)
            st.markdown(f"<a class='link-btn' href='{yt}' target='_blank'>▶ Listen</a>", unsafe_allow_html=True)

            if song not in users[user_id]["liked"]:
                if st.button("Like", key=f"l{i}"):
                    users[user_id]["liked"].append(song)
                    log_action(user_id, "like", song)
                    st.rerun()
            else:
                if st.button("Remove", key=f"r{i}"):
                    users[user_id]["liked"].remove(song)
                    log_action(user_id, "remove", song)
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# PLAYLIST
# -----------------------------
if user_id:
    st.subheader("Your Collection")

    for s in users[user_id]["liked"]:
        st.markdown(f"• {s}")