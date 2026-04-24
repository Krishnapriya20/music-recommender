import pandas as pd
from sklearn.cluster import KMeans
from utils import load_data, get_features, scale_features

df = load_data("data/songs.csv")

features = get_features(df)
X, scaler = scale_features(features)

kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X)

df.to_csv("models/songs_clustered.csv", index=False)

print("Model trained and saved.")