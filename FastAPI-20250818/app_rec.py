from fastapi import FastAPI, HTTPException, Request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

app = FastAPI()

# Carregar os dados
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# Configuração do Jinja2 para renderizar templates HTML
templates = Jinja2Templates(directory="templates")

# 1. Função Best Seller
def best_seller_recommendations_ratings(ratings, movies, top_n=10):
    movie_stats = ratings.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        num_ratings=('rating', 'count')
    ).reset_index()

    min_ratings = 10
    movie_stats = movie_stats[movie_stats['num_ratings'] >= min_ratings]

    best_sellers = movie_stats.sort_values(by=['avg_rating', 'num_ratings'], ascending=[False, False])
    best_sellers = best_sellers.merge(movies, on='movieId')

    return best_sellers.head(top_n).to_dict(orient='records')

# Endpoint para a página inicial
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Endpoints FastAPI
@app.get("/best-seller/ratings/")
async def get_best_seller_ratings(top_n: int = 10):
    return best_seller_recommendations_ratings(ratings, movies, top_n)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8081)