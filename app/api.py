# app/main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional

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
def explore_page(request: Request, tags: Optional[str] = None):
    """
    Main logic hub for the Explore page.
    Handles both "Hot/Trending" (no tags) and "Personalized" (with tags) scenarios.
    """
    featured = []
    recommended = []

    try:
        if tags:
            # Scenario A: User entered tags
            # Parse tags: "Machine Learning, NLP" -> ["machine_learning", "nlp"]
            tag_list = [t.strip().lower().replace(" ", "_") for t in tags.split(",") if t.strip()]
            
            if tag_list:
                # Create preference dictionary mapping tags to fields and keywords
                # We map to both 'field' and 'keywords' dimensions to be safe/broad
                prefs = {
                    "field": tag_list,
                    "keywords": tag_list
                }
                
                # Build profile and vectorize
                profile_text = build_profile_text(prefs, profile_kw_df)
                
                # If profile_text is empty (no tags matched), fallback to hot
                if not profile_text.strip():
                     recs_df = recommend_hot_articles(articles_df, top_k=10)
                else:
                    v_profile = profile_to_vector(profile_text, vectorizer)
                    # Get top 10 recommendations
                    recs_df = recommend_for_profile(v_profile, X_tfidf, articles_df, top_k=10)
            else:
                 recs_df = recommend_hot_articles(articles_df, top_k=10)
        else:
            # Scenario B: No tags / Empty -> Show Hot Articles
            recs_df = recommend_hot_articles(articles_df, top_k=10)

        # Split Strategy
        # Featured: Top 1-5
        featured = recs_df.head(5).to_dict(orient="records")
        
        # Recommended: Top 6-10 (if available)
        if len(recs_df) > 5:
            recommended = recs_df.iloc[5:].to_dict(orient="records")
        else:
            recommended = []

    except Exception as e:
        print(f"Error in explore_page: {e}")
        # Fallback to hot articles in case of error
        try:
            recs_df = recommend_hot_articles(articles_df, top_k=10)
            featured = recs_df.head(5).to_dict(orient="records")
            recommended = recs_df.iloc[5:].to_dict(orient="records")
        except Exception:
            featured = []
            recommended = []

    # Render template
    return templates.TemplateResponse("explore.html", {
        "request": request,
        "featured": featured,
        "recommended": recommended,
        "current_tags": tags or ""
    })


@app.get("/article/{article_id}", response_class=HTMLResponse)
def article_page(article_id: str, request: Request):
    # Find the article in the dataframe
    # Check if ID exists
    article_row = articles_df[articles_df["id"].astype(str) == article_id]
    
    if article_row.empty:
        raise HTTPException(status_code=404, detail="Article not found")
    
    article_data = article_row.iloc[0].to_dict()
    
    # Get similar articles
    try:
        recs = recommend_similar_to_article(article_id, X_tfidf, articles_df, top_k=5)
        recs_list = recs.to_dict(orient="records")
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        recs_list = []

    # Page détail article
    return templates.TemplateResponse("article.html", {
        "request": request, 
        "article": article_data,
        "recommendations": recs_list
    })


# ---------- API JSON ----------

class ProfileRequest(BaseModel):
    prefs: dict
    liked_ids: list[str] = []


class LikeRequest(BaseModel):
    article_id: str
    tags: Optional[str] = None


@app.post("/api/interact/like")
def api_interact_like(req: LikeRequest):
    # 1. Build base profile from tags
    if req.tags:
        tag_list = [t.strip().lower().replace(" ", "_") for t in req.tags.split(",") if t.strip()]
        prefs = {"field": tag_list, "keywords": tag_list}
        profile_text = build_profile_text(prefs, profile_kw_df)
    else:
        profile_text = ""

    # 2. Vectorize base profile
    # If profile_text is empty, vectorizer.transform([""]) returns a zero vector, which is what we want
    v_profile = vectorizer.transform([profile_text])

    # 3. Update with the new like
    # We treat this single like as an update to the session profile
    v_updated = update_profile_with_likes(v_profile, [req.article_id], X_tfidf, articles_df)

    # 4. Recommend (exclude the liked article)
    recs = recommend_for_profile(v_updated, X_tfidf, articles_df, top_k=5, exclude_ids={req.article_id})
    
    return recs.to_dict(orient="records")


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
    # Filter for titles containing the query (case-insensitive)
    df = articles_df[articles_df["title"].str.contains(q, case=False, na=False)]
    
    # Return top 10 results with specific fields
    results = df[["id", "title", "author", "field"]].head(10).to_dict(orient="records")
    return results


@app.get("/api/tags")
def get_tags():
    """
    Returns a list of formatted tags from the profile keywords CSV.
    Example: ["Machine Learning", "Deep Learning", ...]
    """
    if "option" in profile_kw_df.columns:
        # Get unique options, replace underscores with spaces, and title case
        raw_options = profile_kw_df["option"].dropna().unique()
        formatted_tags = [opt.replace("_", " ").title() for opt in raw_options]
        return sorted(list(set(formatted_tags)))
    return []
