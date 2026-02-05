import pandas as pd
import re
import json
from pathlib import Path
from typing import Mapping, Any
import src.config as cfg


def _to_direct_gdrive_url(url: str) -> str:
    # Supports:
    # - https://drive.google.com/file/d/<ID>/view?...
    # - https://drive.google.com/open?id=<ID>
    # - https://drive.google.com/uc?id=<ID>&...
    m = re.search(r"/file/d/([^/]+)", url)
    if not m:
        m = re.search(r"[?&]id=([^&]+)", url)
    if not m:
        return url
    file_id = m.group(1)
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def load_articles(path=None):
    if path is None:
        path = cfg.ARTICLES_PATH

    # Construction du lien de download direct (if lien gdrive)
    if isinstance(path, str) and path.lower().startswith(("http://", "https://", "http:\\", "https:\\")):
        url = path.replace("\\", "/")
        url = _to_direct_gdrive_url(url)

        try:
            df = pd.read_csv(url, low_memory=False)
        except Exception:
            try:
                df = pd.read_parquet(url)
            except Exception as e:
                raise ValueError(f"Impossible de charger le fichier à partir de l'URL: {url}. Erreur: {e}")
    else:
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(path, low_memory=False)
        elif suffix in (".parquet", ".pq"):
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Format non supporté: {suffix} (attendu: .csv ou .parquet)")

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
        path = cfg.PROFILE_KEYWORDS_PATH
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