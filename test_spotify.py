import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="e30998bcf0a44f379a8bc7d5d8fac0e6",
    client_secret="9562cab04ed845fb991f8f53f6d6ddae"
))

results = sp.search(q="Blinding Lights The Weeknd", type="track", limit=1)

print(results)