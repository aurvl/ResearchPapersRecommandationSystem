from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_STOP_WORDS

def fit_vectorizer(corpus):
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        stop_words=TFIDF_STOP_WORDS,
    )
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X
