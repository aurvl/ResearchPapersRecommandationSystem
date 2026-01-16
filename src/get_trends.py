import re
import requests
import json
import joblib
import os
import hashlib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from datetime import datetime, timedelta
from xml.etree import ElementTree as ET
from pathlib import Path
from src.config import ARXIV_API_URL, DATA_CACHE_DIR, HOT_TERMS_NUMB, HOT_TERMS_CACHE_PATH
from nltk.corpus import stopwords

def simple_tokenize(text: str) -> list[str]:
    text = text.lower()
    # virer ce qui n'est pas str
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    tokens = text.split()
    # mots vides minimalistes
    sw = stopwords.words("english")
    tokens = [t for t in tokens if len(t) > 3 and t not in sw]
    return tokens


def get_trends_from_corpus(
    articles_df: pd.DataFrame,
    n_terms: int = 10,
    recent_years: int = 3,
) -> list[str]:
    """
    Extrait des 'mots tendances' a partir des articles les plus recents.
    """
    df = articles_df.copy()
    max_year = df["year"].max()
    cutoff = max_year - recent_years + 1

    recent = df[df["year"] >= cutoff]
    if "text" not in recent.columns:
        recent["text"] = recent["title"] + " " + recent["abstract"]

    counter = Counter()
    for txt in recent["text"]:
        counter.update(simple_tokenize(str(txt)))

    # top n terms les plus frequents
    trends = [w for (w, _) in counter.most_common(n_terms)]
    return trends

def _get_previous_week_dates():
    """
    Returns (start_date, end_date) for the previous full week (Monday to Sunday).
    """
    today = datetime.utcnow()
    # 0 = Monday, 6 = Sunday
    days_since_monday = today.weekday()
    monday_of_current_week = today - timedelta(days=days_since_monday)
    
    # Previous Monday
    start_date = monday_of_current_week - timedelta(days=7)
    # Previous Sunday
    end_date = start_date + timedelta(days=6)
    
    return start_date, end_date

def _get_cache_path_for_previous_week() -> Path:
    start_date, _ = _get_previous_week_dates()
    
    # Naming based on the Monday of that week
    year = start_date.year
    month = start_date.month
    week_of_month = (start_date.day - 1) // 7 + 1
    
    cache_trend_name = f"arxiv_trends_{year}_{month}_w{week_of_month}.json"
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_CACHE_DIR / cache_trend_name

def _load_trends_from_cache() -> list[str]:
    path = _get_cache_path_for_previous_week()
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # on s assure que c est une liste de str
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception as e:
            print("Erreur lecture cache tendances:", e)
    return []


def _save_trends_to_cache(trends: list[str]) -> None:
    path = _get_cache_path_for_previous_week()
    try:
        DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(trends, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Erreur ecriture cache tendances:", e)


def get_trending_from_arxiv(max_results=10):
    """
    Retourne une liste de titres (ou keywords) d'articles recents sur arXiv.

    Tarhet : semaine précédente complet (Lundi à Dimanche).
    """
    start_dt, end_dt = _get_previous_week_dates()

    start_date = start_dt.strftime("%Y%m%d")
    end_date = end_dt.strftime("%Y%m%d")

    query = f"submittedDate:[{start_date}0000 TO {end_date}2359]"

    params = {
        "search_query": query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": max_results,
    }

    try:
        response = requests.get(ARXIV_API_URL, params=params, timeout=10)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        entries = root.findall("{http://www.w3.org/2005/Atom}entry")

        if not entries:
            return []

        # Retourner les TITRES comme "trends"
        trending_terms = []
        for entry in entries:
            title = entry.findtext("{http://www.w3.org/2005/Atom}title", "").strip()
            abstract = entry.findtext("{http://www.w3.org/2005/Atom}summary", "").strip()
            # on combine title + abstract
            text = f"{title} {abstract}"
            trending_terms.append(text)

        return trending_terms

    except Exception as e:
        print("Erreur arXiv API:", e)
        return []


def get_hot_terms(articles_df: pd.DataFrame, top_n=10) -> list[str]:
    """
    Ordre:
      1) on essaie de charger depuis le cache du jour
      2) si vide: on appelle arxiv, on met en cache
      3) si arxiv ne renvoie rien: fallback -> trends du corpus
    """

    # 1. Essayer le cache
    cached = _load_trends_from_cache()
    if cached:
        print("[INFO] Trends charges depuis le cache")
        return cached[:top_n]

    # 2. Essayer arxiv
    arxiv_terms = get_trending_from_arxiv(max_results=top_n)
    if arxiv_terms:
        print("[INFO] Trends source = arxiv API (mise en cache)")
        _save_trends_to_cache(arxiv_terms)
        return arxiv_terms[:top_n]

    # 3. Fallback corpus
    print("[WARN] arxiv indisponible -> trends corpus (non mis en cache)")
    return get_trends_from_corpus(articles_df, n_terms=top_n, recent_years=3)


def _hash_docs(docs: list[str]) -> str:
    h = hashlib.sha256()
    for d in docs:
        h.update((d or "").strip().encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()

def get_hot_term_list(trend_docs: list[str], top_n: int | None = None, renew_cache: bool = True) -> list[str]:
    """Extrait une liste de termes/expressions hot depuis des trend docs (cache disque)."""
    if top_n is None:
        top_n = HOT_TERMS_NUMB

    key = _hash_docs(trend_docs)
    cache_path = str(HOT_TERMS_CACHE_PATH)

    if (not renew_cache) and os.path.exists(cache_path):
        try:
            payload = joblib.load(cache_path)
            if payload.get("key") == key and isinstance(payload.get("terms"), list):
                terms_cached = payload.get("terms", [])
                # si le cache contient assez de termes, on sert direct
                if len(terms_cached) >= top_n:
                    return terms_cached[:top_n]
                # sinon on recalcule (cache trop petit)
        except Exception:
            pass

    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
        norm="l2",
    )
    Td = vec.fit_transform(trend_docs)
    Td = csr_matrix(Td, dtype=np.float32)
    scores = (np.asarray(Td.sum(axis=0)).ravel() / Td.shape[0]).astype(float) # type: ignore
    feats = vec.get_feature_names_out()
    idx = np.argsort(scores)[::-1][:top_n]
    terms: list[str] = [str(feats[int(i)]) for i in idx.tolist()]

    dirpath = os.path.dirname(cache_path)
    if dirpath:  # eviter makedirs("") qd cache_path has no directory component
        os.makedirs(dirpath, exist_ok=True)

    joblib.dump({"key": key, "terms": terms}, cache_path)
    return terms


if __name__ == "__main__":
    # petit test
    from src.data_loading import load_articles
    from src.config import DATA_CACHE_DIR
    df = load_articles()
    terms = get_hot_terms(df, top_n=10)
    print("HOT TERMS:")
    for i, t in enumerate(terms, 1):
        print(f"{i:02d}. {t}")
    
    # On prend quelques articles récents et on combine title + abstract.
    trend_path = DATA_CACHE_DIR / "arxiv_trends_2026_1_w1.json"

    with trend_path.open("r", encoding="utf-8") as f:
        trend_docs = json.load(f)

    assert isinstance(trend_docs, list)
    trend_docs = [str(x) for x in trend_docs]

    print(f"\n[TEST] Loaded {len(trend_docs)} trend_docs from {trend_path.name}")
    terms1 = get_hot_term_list(trend_docs, top_n=10, renew_cache=True)
    print("\nHOT TERMS (from trend_docs):")
    for i, t in enumerate(terms1, 1):
        print(f"{i:02d}. {t}")
    
    # Re-test pour vérifier que le cache joblib marche
    print("\n[TEST] Second call (cache attendu)")
    terms2 = get_hot_term_list(trend_docs, top_n=10, renew_cache=False)
    assert terms1 == terms2
    print("OK: cache hit + same results")