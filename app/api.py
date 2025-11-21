# app/main.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.data_loading import load_articles, load_profile_keywords
from src.text_vectorizer import fit_vectorizer
from src.profile_builder import build_profile_text, profile_to_vector
from src.recommender import (
    recommend_for_profile,
    recommend_hot_articles,
    recommend_similar_to_article,
    update_profile_with_likes,
)

app = FastAPI()

# Fichiers statiques (CSS/JS)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Templates HTML
templates = Jinja2Templates(directory="app/templates")

# Chargement des données en mémoire (une fois au démarrage)
articles_df = load_articles()
profile_kw_df = load_profile_keywords()
vectorizer, X_tfidf = fit_vectorizer(articles_df["text"])


# ---------- PAGES HTML ----------

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    # Page des préférences (index.html)
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/explore", response_class=HTMLResponse)
def explore_page(request: Request):
    # Page principale (reco + recherche)
    return templates.TemplateResponse("explore.html", {"request": request})


@app.get("/article/{article_id}", response_class=HTMLResponse)
def article_page(article_id: str, request: Request):
    # Page détail article
    return templates.TemplateResponse("article.html", {"request": request, "article_id": article_id})


# ---------- API JSON ----------

class ProfileRequest(BaseModel):
    prefs: dict
    liked_ids: list[str] = []


@app.post("/api/recommend/profile")
def api_recommend_profile(req: ProfileRequest):
    profile_text = build_profile_text(req.prefs, profile_kw_df)
    v_profile = profile_to_vector(profile_text, vectorizer)

    if req.liked_ids:
        v_profile = update_profile_with_likes(
            v_profile, req.liked_ids, X_tfidf, articles_df
        )

    recs = recommend_for_profile(v_profile, X_tfidf, articles_df, top_k=5)
    return recs.to_dict(orient="records")


@app.get("/api/recommend/hot")
def api_recommend_hot(top_k: int = 5):
    recs = recommend_hot_articles(articles_df, top_k)
    return recs.to_dict(orient="records")


@app.get("/api/recommend/similar/{article_id}")
def api_recommend_similar(article_id: str, top_k: int = 5):
    recs = recommend_similar_to_article(article_id, X_tfidf, articles_df, top_k)
    return recs.to_dict(orient="records")


@app.get("/api/search")
def api_search(q: str):
    df = articles_df[
        articles_df["title"].str.contains(q, case=False)
        | articles_df["abstract"].str.contains(q, case=False)
    ]
    return df.head(10).to_dict(orient="records")
