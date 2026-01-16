from contextlib import asynccontextmanager
import json
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
import numpy as np

from src.data_loading import load_articles, load_profile_keywords, json_to_text
from src.profile_builder import build_profile_text
from src.utils import get_article_image
from src.embeddings import get_or_compute_article_embeddings, encode_texts
from src.recommender import (
    recommend_for_profile,
    recommend_hot_articles,
    recommend_similar_to_article,
    update_profile_with_likes,
    recommend_for_query_text,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    articles_df = load_articles()
    profile_kw_df = load_profile_keywords()

    # Load/cache SentenceTransformer model + (N, 384) embeddings once.
    E_articles, model = get_or_compute_article_embeddings(articles_df, text_col="text")

    app.state.articles_df = articles_df
    app.state.profile_kw_df = profile_kw_df
    app.state.E_articles = E_articles
    app.state.model = model
    app.state.id_str_to_id = dict(zip(articles_df["id"].astype(str), articles_df["id"]))

    yield

app = FastAPI(lifespan=lifespan)

# Fichiers statiques (CSS/JS)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Templates HTML
templates = Jinja2Templates(directory="app/templates")

def _resolve_catalog_id(article_id: str):
    return app.state.id_str_to_id.get(str(article_id))


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
                
                # Build profile text and embed it
                profile_text = build_profile_text(prefs, app.state.profile_kw_df)
                
                # If profile_text is empty (no tags matched), fallback to hot
                if not profile_text.strip():
                    recs_df = recommend_hot_articles(
                        app.state.articles_df,
                        method="semantic",
                        top_k=10,
                        model=app.state.model,
                        renew=False,
                        E_articles=app.state.E_articles,
                    )
                else:
                    v_profile = encode_texts(
                        [profile_text],
                        model=app.state.model,
                        normalize=True,
                        show_progress_bar=False,
                    )
                    recs_df = recommend_for_profile(v_profile, app.state.E_articles, app.state.articles_df, top_k=10)
            else:
                recs_df = recommend_hot_articles(
                    app.state.articles_df,
                    method="semantic",
                    top_k=10,
                    model=app.state.model,
                    renew=False,
                    E_articles=app.state.E_articles,
                )
        else:
            # Scenario B: No tags / Empty -> Show Hot Articles
            recs_df = recommend_hot_articles(
                app.state.articles_df,
                method="semantic",
                top_k=10,
                model=app.state.model,
                renew=False,
                E_articles=app.state.E_articles,
            )

        # Split Strategy
        # Featured: Top 1-5
        featured = recs_df.head(5).to_dict(orient="records")
        for f in featured:
            f["image_url"] = get_article_image(f.get("field"))
        
        # Recommended: Top 6-10 (if available)
        if len(recs_df) > 5:
            recommended = recs_df.iloc[5:].to_dict(orient="records")
            for r in recommended:
                r["image_url"] = get_article_image(r.get("field"))
        else:
            recommended = []

    except Exception as e:
        print(f"Error in explore_page: {e}")
        # Fallback to hot articles in case of error
        try:
            recs_df = recommend_hot_articles(
                app.state.articles_df,
                method="semantic",
                top_k=10,
                model=app.state.model,
                renew=False,
                E_articles=app.state.E_articles,
            )
            featured = recs_df.head(5).to_dict(orient="records")
            for f in featured:
                f["image_url"] = get_article_image(f.get("field"))
            
            recommended = recs_df.iloc[5:].to_dict(orient="records")
            for r in recommended:
                r["image_url"] = get_article_image(r.get("field"))
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
    articles_df = app.state.articles_df
    article_row = articles_df[articles_df["id"].astype(str) == article_id]
    
    if article_row.empty:
        raise HTTPException(status_code=404, detail="Article not found")
    
    article_data = article_row.iloc[0].to_dict()
    article_data["image_url"] = get_article_image(article_data.get("field"))
    
    # Get similar articles
    try:
        catalog_id = article_data.get("id")
        recs = recommend_similar_to_article(catalog_id, app.state.E_articles, articles_df, top_k=5)
        recs_list = recs.to_dict(orient="records")
        for r in recs_list:
            r["image_url"] = get_article_image(r.get("field"))
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
        profile_text = build_profile_text(prefs, app.state.profile_kw_df)
    else:
        profile_text = ""

    # 2. Embed base profile
    if profile_text.strip():
        v_profile = encode_texts(
            [profile_text],
            model=app.state.model,
            normalize=True,
            show_progress_bar=False,
        )
    else:
        v_profile = np.zeros((1, app.state.E_articles.shape[1]), dtype=np.float32)

    # 3. Update with the new like
    # We treat this single like as an update to the session profile
    liked_id = _resolve_catalog_id(req.article_id)
    if liked_id is None:
        raise HTTPException(status_code=404, detail="Article not found")
    v_updated = update_profile_with_likes(v_profile, [liked_id], app.state.E_articles, app.state.articles_df, desparse=True)

    # 4. Recommend (exclude the liked article)
    recs = recommend_for_profile(
        v_updated,
        app.state.E_articles,
        app.state.articles_df,
        top_k=5,
        exclude_ids={liked_id},
    )
    
    results = recs.to_dict(orient="records")
    for r in results:
        r["image_url"] = get_article_image(r.get("field"))
    return results


@app.post("/api/recommend/profile")
def api_recommend_profile(req: ProfileRequest):
    profile_text = build_profile_text(req.prefs, app.state.profile_kw_df)
    if profile_text.strip():
        v_profile = encode_texts(
            [profile_text],
            model=app.state.model,
            normalize=True,
            show_progress_bar=False,
        )
    else:
        v_profile = np.zeros((1, app.state.E_articles.shape[1]), dtype=np.float32)

    if req.liked_ids:
        liked_ids = [x for x in (_resolve_catalog_id(i) for i in req.liked_ids) if x is not None]
        v_profile = update_profile_with_likes(v_profile, liked_ids, app.state.E_articles, app.state.articles_df, desparse=True)

    recs = recommend_for_profile(v_profile, app.state.E_articles, app.state.articles_df, top_k=5)
    results = recs.to_dict(orient="records")
    for r in results:
        r["image_url"] = get_article_image(r.get("field"))
    return results


@app.get("/api/recommend/hot")
def api_recommend_hot(top_k: int = 5):
    recs = recommend_hot_articles(
        app.state.articles_df,
        method="semantic",
        top_k=top_k,
        model=app.state.model,
        renew=False,
        E_articles=app.state.E_articles,
    )
    results = recs.to_dict(orient="records")
    for r in results:
        r["image_url"] = get_article_image(r.get("field"))
    return results


@app.get("/api/recommend/similar/{article_id}")
def api_recommend_similar(article_id: str, top_k: int = 5):
    catalog_id = _resolve_catalog_id(article_id)
    if catalog_id is None:
        raise HTTPException(status_code=404, detail="Article not found")
    recs = recommend_similar_to_article(catalog_id, app.state.E_articles, app.state.articles_df, top_k)
    results = recs.to_dict(orient="records")
    for r in results:
        r["image_url"] = get_article_image(r.get("field"))
    return results


@app.get("/api/search")
def api_search(q: str):
    # Filter for titles containing the query (case-insensitive)
    df = app.state.articles_df[app.state.articles_df["title"].str.contains(q, case=False, na=False)]
    
    # Return top 10 results with specific fields
    results = df[["id", "title", "author", "field"]].head(10).to_dict(orient="records")
    for r in results:
        r["image_url"] = get_article_image(r.get("field"))
    return results


@app.get("/api/tags")
def get_tags():
    """
    Returns a list of formatted tags from the profile keywords CSV.
    Example: ["Machine Learning", "Deep Learning", ...]
    """
    if "option" in app.state.profile_kw_df.columns:
        # Get unique options, replace underscores with spaces, and title case
        raw_options = app.state.profile_kw_df["option"].dropna().unique()
        formatted_tags = [opt.replace("_", " ").title() for opt in raw_options]
        return sorted(list(set(formatted_tags)))
    return []


# ---------- External JSON (ephemeral upload) ----------

@app.get("/external", response_class=HTMLResponse)
def external_page(request: Request):
    return templates.TemplateResponse("external.html", {
        "request": request,
        "error": None,
        "uploaded": None,
        "recommendations": [],
    })


@app.post("/external", response_class=HTMLResponse)
async def external_upload(request: Request, file: UploadFile = File(...)):
    error = None
    uploaded = None
    recs_list = []

    try:
        if not file.filename.lower().endswith(".json"):
            raise ValueError("Please upload a .json file")

        raw = await file.read()
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON must be an object (dictionary)")

        query_text = json_to_text(payload)
        if not query_text.strip():
            raise ValueError("The uploaded JSON does not contain enough text (need at least title/abstract/field)")

        uploaded = {
            "title": (payload.get("title") or "").strip(),
            "abstract": (payload.get("abstract") or payload.get("summary") or "").strip(),
            "field": (payload.get("field") or "").strip(),
            "url": (payload.get("url") or "").strip(),
        }

        recs_df = recommend_for_query_text(
            query_text,
            "embed",
            app.state.E_articles,
            app.state.articles_df,
            top_k=10,
            model=app.state.model,
        )
        recs_list = recs_df.to_dict(orient="records")
        for r in recs_list:
            r["image_url"] = get_article_image(r.get("field"))

    except Exception as e:
        error = str(e)

    return templates.TemplateResponse("external.html", {
        "request": request,
        "error": error,
        "uploaded": uploaded,
        "recommendations": recs_list,
    })
