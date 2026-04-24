import streamlit as st
import pandas as pd
from utils import scale_features, get_features, recommend

df = pd.read_csv("models/songs_clustered.csv")

features = get_features(df)
X, _ = scale_features(features)

st.title("Context-Aware Music Recommender System")

mood_profiles = {
    "Happy": [0.9, 0.9, 120],
    "Sad": [0.2, 0.3, 60],
    "Chill": [0.4, 0.5, 70],
    "Energetic": [0.9, 0.6, 150]
}

mood = st.selectbox("Select mood", list(mood_profiles.keys()))

if st.button("Recommend"):
    mood_vector = [mood_profiles[mood]]
    results = recommend(mood_vector, X, df)
    st.write(results["track_name"].head(10).tolist())