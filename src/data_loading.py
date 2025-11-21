import pandas as pd
from src.config import ARTICLES_PATH, PROFILE_KEYWORDS_PATH

def load_articles(path=None):
    if path is None:
        path = ARTICLES_PATH
    df = pd.read_csv(path)
    df = df.dropna(subset=["title", "abstract"]).reset_index(drop=True)
    df["text"] = df["title"] + " " + df["abstract"]
    return df

def load_profile_keywords(path=None):
    if path is None:
        path = PROFILE_KEYWORDS_PATH
    return pd.read_csv(path)
