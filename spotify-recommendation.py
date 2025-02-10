import requests
import base64
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import spotipy
from spotipy.oauth2 import SpotifyOAuth


CLIENT_ID = 'your_client_id'
CLIENT_SECRET = 'your_client_secret'


def get_access_token(client_id, client_secret):
    client_credentials = f"{client_id}:{client_secret}"
    client_credentials_base64 = base64.b64encode(client_credentials.encode()).decode()
    token_url = 'https://accounts.spotify.com/api/token'
    headers = {'Authorization': f'Basic {client_credentials_base64}'}
    data = {'grant_type': 'client_credentials'}
    response = requests.post(token_url, data=data, headers=headers)

    if response.status_code == 200:
        return response.json()['access_token']
    else:
        raise Exception("Error obtaining access token. Check your client credentials.")


def get_trending_playlist_data(playlist_id, access_token):
    sp = spotipy.Spotify(auth=access_token)

    
    playlist_tracks = sp.playlist_tracks(playlist_id, fields='items(track(id, name, artists, album(id, name)))')

    music_data = []
    for track_info in playlist_tracks['items']:
        track = track_info['track']
        if not track:
            continue

        track_name = track['name']
        artists = ', '.join([artist['name'] for artist in track['artists']])
        album_name = track['album']['name']
        album_id = track['album']['id']
        track_id = track['id']

        # Fetch audio features
        audio_features = sp.audio_features(track_id)[0] if track_id else None

        # Fetch album release date
        album_info = sp.album(album_id) if album_id else None
        release_date = album_info['release_date'] if album_info else None

        # Fetch track popularity
        track_details = sp.track(track_id) if track_id else None
        popularity = track_details['popularity'] if track_details else None

        # Track details
        track_data = {
            'Track Name': track_name,
            'Artists': artists,
            'Album Name': album_name,
            'Album ID': album_id,
            'Track ID': track_id,
            'Popularity': popularity,
            'Release Date': release_date,
            'Duration (ms)': audio_features['duration_ms'] if audio_features else None,
            'Danceability': audio_features['danceability'] if audio_features else None,
            'Energy': audio_features['energy'] if audio_features else None,
            'Key': audio_features['key'] if audio_features else None,
            'Loudness': audio_features['loudness'] if audio_features else None,
            'Mode': audio_features['mode'] if audio_features else None,
            'Speechiness': audio_features['speechiness'] if audio_features else None,
            'Acousticness': audio_features['acousticness'] if audio_features else None,
            'Instrumentalness': audio_features['instrumentalness'] if audio_features else None,
            'Liveness': audio_features['liveness'] if audio_features else None,
            'Valence': audio_features['valence'] if audio_features else None,
            'Tempo': audio_features['tempo'] if audio_features else None,
        }

        music_data.append(track_data)

    return pd.DataFrame(music_data)

# Calculate weighted popularity
def calculate_weighted_popularity(release_date):
    try:
        release_date = datetime.strptime(release_date, '%Y-%m-%d')
        time_span = (datetime.now() - release_date).days
        return 1 / (time_span + 1)
    except Exception:
        return 0

# Content-based recommendations
def content_based_recommendations(input_song_name, music_df, music_features_scaled, num_recommendations=5):
    if input_song_name not in music_df['Track Name'].values:
        print(f"'{input_song_name}' not found in the dataset.")
        return None

    input_song_index = music_df[music_df['Track Name'] == input_song_name].index[0]
    similarity_scores = cosine_similarity([music_features_scaled[input_song_index]], music_features_scaled)
    similar_song_indices = similarity_scores.argsort()[0][::-1][1:num_recommendations + 1]

    return music_df.iloc[similar_song_indices][['Track Name', 'Artists', 'Album Name', 'Release Date', 'Popularity']]

# Hybrid recommendations
def hybrid_recommendations(input_song_name, music_df, music_features_scaled, num_recommendations=5):
    if input_song_name not in music_df['Track Name'].values:
        print(f"'{input_song_name}' not found in the dataset.")
        return None

    content_based_rec = content_based_recommendations(input_song_name, music_df, music_features_scaled, num_recommendations)
    weighted_popularity = calculate_weighted_popularity(
        music_df.loc[music_df['Track Name'] == input_song_name, 'Release Date'].values[0]
    )
    return content_based_rec.sort_values(by='Popularity', ascending=False)


if __name__ == "__main__":
    try:
        access_token = get_access_token(CLIENT_ID, CLIENT_SECRET)
        playlist_id = "playlist-id"   
        music_df = get_trending_playlist_data(playlist_id, access_token)

        # Normalize features
        scaler = MinMaxScaler()
        music_features = music_df[['Danceability', 'Energy', 'Key', 'Loudness', 'Mode',
                                   'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']].fillna(0).values
        music_features_scaled = scaler.fit_transform(music_features)

        # Recommendations
        input_song_name = "I'm Good (Blue)"
        recommendations = hybrid_recommendations(input_song_name, music_df, music_features_scaled)
        print(f"Hybrid recommended songs for '{input_song_name}':")
        print(recommendations)

    except Exception as e:
        print(f"Error: {e}")
