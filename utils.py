import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def load_data(path):
    return pd.read_csv(path)

def get_features(df):
    return df[["energy", "valence", "tempo"]]

def scale_features(features):
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    return X, scaler

def recommend(mood_vector, X, df):
    similarity = cosine_similarity(mood_vector, X)
    df["score"] = similarity[0]
    return df.sort_values("score", ascending=False)