# Spotify-Recommendation

This project is a Python-based recommendation system that retrieves data from Spotify's Web API and provides song recommendations based on audio features and popularity inspired from a tutorial.

## Features

- **Playlist Data Retrieval**: Fetches tracks, audio features, and metadata (e.g., popularity, release date) from a specified Spotify playlist.
- **Content-Based Recommendations**: Suggests songs similar to a given track using cosine similarity on audio features.
- **Hybrid Recommendations**: Combines content-based filtering with a weighted popularity score based on release date.
- **Feature Scaling**: Normalizes audio features for better similarity computations.

---

## Requirements

- Python 3.x
- Spotify Developer Account
- Spotify Playlist ID
- Python Libraries:
  - `requests`
  - `spotipy`
  - `numpy`
  - `pandas`
  - `scikit-learn`
