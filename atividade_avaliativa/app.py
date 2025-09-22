from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()

# CARREGANDO O CSV
df = pd.read_csv("top50MusicFrom2010-2019.csv")

# RENOMEANDO COLUNAS (ajustar se houver diferença no seu CSV real)
df = df.rename(columns={
    'title': 'Title',
    'artist': 'Artist',
    'the genre of the track': 'Genre',
    'year': 'Year',
    'Beats.Per.Minute -The tempo of the song': 'Beats.Per.Minute',
    'Energy- The energy of a song - the higher the value, the more energtic': 'Energy',
    'Danceability - The higher the value, the easier it is to dance to this song': 'Danceability',
    'Loudness/dB - The higher the value, the louder the song': 'Loudness',
    'Liveness - The higher the value, the more likely the song is a live recording': 'Liveness',
    'Valence - The higher the value, the more positive mood for the song': 'Valence',
    'Length - The duration of the song': 'Length',
    'Acousticness - The higher the value the more acoustic the song is': 'Acousticness',
    'Speechiness - The higher the value the more spoken word the song contains': 'Speechiness',
    'Popularity- The higher the value the more popular the song is': 'Popularity'
})

# FEATURES NUMÉRICAS
features = ['Beats.Per.Minute', 'Energy', 'Danceability', 'Loudness',
            'Liveness', 'Valence', 'Length', 'Acousticness',
            'Speechiness', 'Popularity']

scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# MATRIZ DE SIMILARIDADE
similarity_matrix = cosine_similarity(df[features])

# MODELOS DE REQUEST
class GenreArtistRequest(BaseModel):
    genre: Optional[str] = None
    artist: Optional[str] = None
    limit: int = 5

class HybridRequest(BaseModel):
    song_title: str
    user_id: str
    content_weight: float = 0.7
    collab_weight: float = 0.3
    limit: int = 5

# ENDPOINTS
@app.get("/recommendations/content-based/{song_title}")
async def content_based_recommendations(song_title: str, limit: int = 5):
    if song_title not in df['Title'].values:
        return {"error": "Música não encontrada"}

    idx = df[df['Title'] == song_title].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:limit+1]

    recommendations = df.iloc[[i[0] for i in sim_scores]][['Title', 'Artist', 'Genre']]
    return recommendations.to_dict(orient="records")


@app.post("/recommendations/genre-artist")
async def genre_artist_recommendations(request: GenreArtistRequest):
    result = df
    if request.genre:
        result = result[result['Genre'].str.contains(request.genre, case=False, na=False)]
    if request.artist:
        result = result[result['Artist'].str.contains(request.artist, case=False, na=False)]

    recommendations = result[['Title', 'Artist', 'Genre']].head(request.limit)
    return recommendations.to_dict(orient="records")


@app.get("/recommendations/collaborative/{user_id}")
async def collaborative_recommendations(user_id: str):
    # SUGERINDO MÚSICAS POPULARES
    recommendations = df.sort_values(by="Popularity", ascending=False)[['Title', 'Artist', 'Genre']].head(5)
    return recommendations.to_dict(orient="records")


@app.post("/recommendations/hybrid")
async def hybrid_recommendations(request: HybridRequest):
    if request.song_title not in df['Title'].values:
        return {"error": "Música não encontrada"}

    # CONTENT BASED
    idx = df[df['Title'] == request.song_title].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    content_scores = {i[0]: i[1] for i in sim_scores[1:request.limit*2]}

    # SIMULADO PELA POPULARIDADE
    collab_scores = df['Popularity'] / 100.0

    # PARTE HÍBRIDA
    hybrid_scores = {}
    for i in content_scores:
        hybrid_scores[i] = (request.content_weight * content_scores[i]) + \
                           (request.collab_weight * collab_scores.iloc[i])

    sorted_scores = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sorted_scores[:request.limit]]

    recommendations = df.iloc[top_indices][['Title', 'Artist', 'Genre']]
    return recommendations.to_dict(orient="records")


@app.get("/recommendations/popular")
async def popular_recommendations(year: Optional[int] = None, genre: Optional[str] = None, limit: int = 5):
    result = df
    if year and 'Year' in df.columns:
        result = result[result['Year'] == year]
    if genre:
        result = result[result['Genre'].str.contains(genre, case=False, na=False)]

    recommendations = result.sort_values(by="Popularity", ascending=False)[['Title', 'Artist', 'Genre']].head(limit)
    return recommendations.to_dict(orient="records")

# TESTANDO
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=True)
