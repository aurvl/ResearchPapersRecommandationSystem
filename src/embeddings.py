import os
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

def get_embedding_model(model_name: str):
    """
    Load and return a SentenceTransformer model.

    Args:
        model_name (str): Name of the Hugging Face model to load.

    Returns:
        SentenceTransformer: Loaded model.
    """
    return SentenceTransformer(model_name)

def encode_texts(texts: list[str], model, batch_size=64, normalize=True):
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
        show_progress_bar=True,
        convert_to_numpy=True
    )
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
    return embeddings.astype(np.float32)

def get_or_compute_article_embeddings(
    articles_df,
    text_col="text",
    cache_path="models/article_embeddings.npy",
    model_name="sentence-transformers/all-MiniLM-L6-v2"
):
    """
    Get or compute embeddings for articles.

    Args:
        articles_df (pd.DataFrame): DataFrame containing articles.
        text_col (str): Column name containing text to encode.
        cache_path (str): Path to save/load cached embeddings.
        model_name (str): Name of the Hugging Face model to use.

    Returns:
        np.ndarray: Embeddings as a numpy array of shape (n, dim).
    """
    if os.path.exists(cache_path):
        print("Loading cached embeddings...")
        cached_data = joblib.load(cache_path)
        if (
            "embeddings" in cached_data
            and "ids" in cached_data
            and len(cached_data["ids"]) == len(articles_df)
            and cached_data["ids"] == articles_df["id"].tolist()
        ):
            return cached_data["embeddings"]

    print("Cache not found or incompatible. Computing embeddings...")
    model = get_embedding_model(model_name)
    embeddings = encode_texts(articles_df[text_col].tolist(), model)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    joblib.dump({"embeddings": embeddings, "ids": articles_df["id"].tolist()}, cache_path)

    return embeddings
