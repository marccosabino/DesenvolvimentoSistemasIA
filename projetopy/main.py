from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# ======================
# 1. Carregar CSV real
# ======================
df = pd.read_csv("top50MusicFrom2010-2019.csv")

# Renomear colunas para nomes simples
df = df.rename(columns={
    'title': 'Title',
    'artist': 'Artist',
    'the genre of the track': 'Genre',
    'year': 'Year',
    'Beats.Per.Minute -The tempo of the song': 'Beats.Per.Minute',
    'Energy- The energy of a song - the higher the value, the more energtic': 'Energy',
    'Danceability - The higher the value, the easier it is to dance to this song': 'Danceability',
    'Loudness/dB - The higher the value, the louder the song': 'Loudness/dB',
    'Liveness - The higher the value, the more likely the song is a live recording': 'Liveness',
    'Valence - The higher the value, the more positive mood for the song': 'Valence',
    'Length - The duration of the song': 'Length',
    'Acousticness - The higher the value the more acoustic the song is': 'Acousticness',
    'Speechiness - The higher the value the more spoken word the song contains': 'Speechiness',
    'Popularity- The higher the value the more popular the song is': 'Popularity'
})

# ======================
# 2. Pré-processar features
# ======================
features = ['Beats.Per.Minute','Energy','Danceability','Loudness/dB','Liveness',
            'Valence','Length','Acousticness','Speechiness','Popularity']
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Similaridade baseada em conteúdo
similarity_matrix = cosine_similarity(df[features])

# ======================
# 3. Simular usuários (Filtro Colaborativo)
# ======================
user_likes = {
    "user1": [df.iloc[0]['Title'], df.iloc[1]['Title']],
    "user2": [df.iloc[2]['Title']],
    "user3": [df.iloc[3]['Title'], df.iloc[4]['Title']],
}

# ======================
# 4. Models pydantic
# ======================
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

# ======================
# 5. Endpoints
# ======================

# Rota raiz
@app.get("/")
def root():
    return {"message": "🎶 API de Recomendação de Músicas ativa! Acesse /docs para explorar."}

# 1. Conteúdo
@app.get("/recommendations/content-based/{song_title}")
async def content_based_recommendations(song_title: str, limit: int = 5):
    idx = df[df['Title']==song_title].index
    if len(idx)==0:
        return {"error":"Song not found"}
    idx = idx[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)[1:limit+1]
    recs = df.iloc[[i[0] for i in sim_scores]][['Title','Artist','Genre']]
    return recs.to_dict(orient='records')

# 2. Gênero/Artista
@app.post("/recommendations/genre-artist")
async def genre_artist_recommendations(request: GenreArtistRequest):
    query = df.copy()
    if request.genre:
        query = query[query['Genre'].str.contains(request.genre, case=False, na=False)]
    if request.artist:
        query = query[query['Artist'].str.contains(request.artist, case=False, na=False)]
    query = query.sort_values('Popularity', ascending=False).head(request.limit)
    return query[['Title','Artist','Genre','Popularity']].to_dict(orient='records')

# 3. Colaborativo
@app.get("/recommendations/collaborative/{user_id}")
async def collaborative_recommendations(user_id: str, limit: int = 5):
    if user_id not in user_interactions:
        return []

    user_songs = set(user_interactions[user_id])
    recommendations = {}

    for other_user, songs in user_interactions.items():
        if other_user == user_id:
            continue
        common = user_songs.intersection(songs)
        if common:
            for song in songs:
                if song not in user_songs:
                    recommendations[song] = recommendations.get(song, 0) + 1

    sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    result = []
    for song, score in sorted_recs[:limit]:
        artist = df[df['Title']==song]['Artist'].values[0]
        genre = df[df['Title']==song]['Genre'].values[0]
        result.append({"Title": song, "Artist": artist, "Genre": genre, "Score": score})
    return result

# 4. Híbrido
@app.post("/recommendations/hybrid")
async def hybrid_recommendations(request: HybridRequest):
    content_recs = await content_based_recommendations(request.song_title, limit=request.limit)
    collab_recs = await collaborative_recommendations(request.user_id)
    content_scores = {r['Title']:request.content_weight for r in content_recs}
    collab_scores = {r['Title']:request.collab_weight for r in collab_recs}
    combined = {}
    for k,v in content_scores.items():
        combined[k]=combined.get(k,0)+v
    for k,v in collab_scores.items():
        combined[k]=combined.get(k,0)+v
    result = sorted(combined.items(), key=lambda x:x[1], reverse=True)[:request.limit]
    return [{"Title":title,"score":score} for title,score in result]

# 5. Popularidade/Ano
@app.get("/recommendations/popular")
async def popular_recommendations(year: Optional[int] = None, genre: Optional[str] = None, limit: int = 5):
    query = df.copy()
    if year:
        query = query[query['Year']==year]
    if genre:
        query = query[query['Genre'].str.contains(genre, case=False, na=False)]
    query = query.sort_values('Popularity', ascending=False).head(limit)
    return query[['Title','Artist','Genre','Popularity','Year']].to_dict(orient='records')

# ======================
# Execução direta
# ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
