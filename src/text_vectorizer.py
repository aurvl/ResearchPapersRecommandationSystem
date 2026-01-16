import os
import joblib
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import (
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_STOP_WORDS,
    TFIDF_NORM,
    TFIDF_VECTORIZER_PATH,
    X_TFIDF_PATH,
)

def fit_vectorizer(corpus):
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        stop_words=TFIDF_STOP_WORDS,
        norm=TFIDF_NORM,
    )
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X

def save_tfidf_elements(vectorizer, X):
    os.makedirs(os.path.dirname(str(TFIDF_VECTORIZER_PATH)), exist_ok=True)
    joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)
    joblib.dump(X, X_TFIDF_PATH)

@lru_cache(maxsize=1)
def load_tfidf_elements():
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    X = joblib.load(X_TFIDF_PATH)
    return vectorizer, X