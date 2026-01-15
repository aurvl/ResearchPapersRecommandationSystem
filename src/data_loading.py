import pandas as pd
import json
from typing import Mapping, Any
from src.config import ARTICLES_PATH, PROFILE_KEYWORDS_PATH

def load_articles(path=None):
    if path is None:
        path = ARTICLES_PATH
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["title", "abstract", "field"]).reset_index(drop=True)
    df["text"] = (
        df[["title", "abstract", "field"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )
    return df

def load_profile_keywords(path=None):
    if path is None:
        path = PROFILE_KEYWORDS_PATH
    return pd.read_csv(path)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def json_to_text(payload: Mapping[str, Any]) -> str:
    """
    Convert a JSON payload to plain text for vectorization.
    Extracts relevant fields (title, abstract, field).
    """
    title = (payload.get("title") or "").strip()
    abstract = (payload.get("abstract") or payload.get("summary") or "").strip()
    field = (payload.get("field") or "").strip()
    authors = payload.get("authors") or ""
    
    if isinstance(authors, list):
        authors = " ".join([str(a) for a in authors])
    authors = str(authors).strip()

    categories = payload.get("categories") or payload.get("tags") or ""
    if isinstance(categories, list):
        categories = " ".join([str(c) for c in categories])
    categories = str(categories).strip()

    text = " ".join([title, abstract, field, authors, categories]).strip()
    return " ".join(text.split())

def query_to_text(path = None) -> str:
    if path is None:
        path = "data/payload.json"
    payload = load_json(path)
    return json_to_text(payload)