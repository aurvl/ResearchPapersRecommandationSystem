from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
from src.config import EMB_PATH, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_STOP_WORDS, LLM_URL

def fit_vectorizer(corpus):
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        stop_words=TFIDF_STOP_WORDS,
    )
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X

def compute_embeddings(texts):
    print("Loading sentence-transformer model...")
    model = SentenceTransformer(LLM_URL)

    print("Encoding articles...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # => dot = cosine
    )
    import os
    os.makedirs(os.path.dirname(EMB_PATH), exist_ok=True)
    np.save(EMB_PATH, embeddings)
    return embeddings
