import os
import numpy as np
import joblib
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from src.config import EMBED_CORPUS, LLM_URL

@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = LLM_URL):
    """
    Load and return a SentenceTransformer model.

    Args:
        model_name (str): Name of the Hugging Face model to load. Defaults to LLM_URL.

    Returns:
        SentenceTransformer: Loaded model.
    """
    return SentenceTransformer(model_name, device="cpu")

def encode_texts(
    texts: list[str],
    model,
    batch_size: int = 64,
    normalize: bool = True,
    show_progress_bar: bool = False,
):
    """
    Encode a list of texts into embeddings.

    Args:
        texts (list[str]): List of texts to encode.
        model (SentenceTransformer): Preloaded SentenceTransformer model.
        batch_size (int): Batch size for encoding.
        normalize (bool): Whether to normalize embeddings with L2 norm.

    Returns:
        np.ndarray: Embeddings as a numpy array of shape (n, dim).
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True, device="cpu"
    )
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        embeddings = embeddings / norms
    return embeddings.astype(np.float32)

def get_or_compute_article_embeddings(
    articles_df,
    text_col="text",
    cache_path=EMBED_CORPUS
):
    """
    Get or compute embeddings for articles.

    Args:
        articles_df (pd.DataFrame): DataFrame containing articles.
        text_col (str): Column name containing text to encode.
        cache_path (str): Path to save/load cached embeddings.

    Returns:
        np.ndarray: Embeddings as a numpy array of shape (n, dim).
        model: Loaded SentenceTransformer model.
    """
    if os.path.exists(cache_path):
        print("Loading cached embeddings...")
        cached_data = joblib.load(cache_path)
        model = get_embedding_model()
        if (
            "embeddings" in cached_data
            and "ids" in cached_data
            and len(cached_data["ids"]) == len(articles_df)
            and cached_data["ids"] == articles_df["id"].tolist()
        ):
            return cached_data["embeddings"], model

    print("Cache not found or incompatible. Computing embeddings...")
    model = get_embedding_model()
    embeddings = encode_texts(
        articles_df[text_col].tolist(),
        model,
        normalize=True,
        show_progress_bar=True,
    )

    # Save to cache
    joblib.dump({"embeddings": embeddings, "ids": articles_df["id"].tolist()}, cache_path)

    return embeddings, model