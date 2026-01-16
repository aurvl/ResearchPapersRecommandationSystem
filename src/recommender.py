import os
import numpy as np
import joblib
from scipy.sparse import csr_matrix, issparse
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import linear_kernel
from src.config import (
    TOP_K_MAIN, TOP_K_SIMILAR, PROFILE_ALPHA, 
    HOT_TERMS_NUMB, HOT_TFIDF_TRENDS_CACHE_PATH
)
from src.utils import print_l2_norm_stats, mean_topk, safe_minmax_norm, topk_indices
from src.text_vectorizer import load_tfidf_elements
from src.embeddings import get_or_compute_article_embeddings, encode_texts
from src.get_trends import get_hot_terms, get_hot_term_list

def _to_unit_sparse(v):
    if not issparse(v):
        v = csr_matrix(v)
    return normalize(v, norm="l2", axis=1, copy=False)

def _cached_trend_tfidf_matrix(vectorizer, trend_docs):
    """Cache la matrice TF-IDF (n_trends, V) transformee avec le vectorizer catalogue."""
    import hashlib
    key = hashlib.sha256("\n".join([str(x) for x in trend_docs]).encode("utf-8")).hexdigest()
    path = str(HOT_TFIDF_TRENDS_CACHE_PATH)

    if os.path.exists(path):
        try:
            payload = joblib.load(path)
            if payload.get("key") == key:
                return payload["T"]
        except Exception:
            pass

    T = vectorizer.transform(trend_docs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"key": key, "T": T}, path)
    return T

def recommend_for_profile(v_profile, X, articles_df, top_k=TOP_K_MAIN, exclude_ids=None):
    if exclude_ids is None:
        exclude_ids = set()
    # Enforce L2-normlz pour que dot product=cosine sim
    v = _to_unit_sparse(v_profile)

    if os.getenv("DEBUG_L2_NORMS") == "1":
        print_l2_norm_stats(v, "v_profile", 1)

    sims = linear_kernel(v, X).ravel()
    excl = articles_df["id"].isin(exclude_ids).values if "id" in articles_df.columns else None
    idx = topk_indices(sims, top_k, exclude_mask=excl)
    return articles_df.iloc[idx]

def update_profile_with_likes(v_profile, liked_ids, X, articles_df, alpha=PROFILE_ALPHA, desparse=False):
    """
    v_profile : vecteur TF-IDF/Embed (1, D) du profil courant
    liked_ids : liste D id d'articles que l'utilisateur a likés
    X   : matrice TF-IDF/Embed des articles
    articles_df : DataFrame des articles
    alpha     : poids du profil initial vs likes (0 <= alpha <= 1)

    Retourne un nouveau vecteur de profil v_new.
    """
    if not liked_ids:
        return v_profile

    mask = articles_df["id"].isin(liked_ids)
    if not mask.any():
        return v_profile

    liked_vecs = X[mask.values]         # vecteurs des articles likés -> sparse (n_liked, D)
    liked_centroid = liked_vecs.mean(axis=0)  # centroide (1, D)

    # assurer CSR : debug
    if not issparse(liked_centroid):
        liked_centroid = csr_matrix(liked_centroid)

    if not issparse(v_profile):
        v_profile = csr_matrix(v_profile)
    
    v_new = alpha * v_profile + (1 - alpha) * liked_centroid
    # Renormalisation
    v_new = normalize(v_new, norm="l2", axis=1, copy=False)

    # debug: print norms pre/post feedback maj.
    import os
    if os.getenv("DEBUG_L2_NORMS") == "1":
        from src.utils import print_l2_norm_stats
        print_l2_norm_stats(v_profile, name="v_profile_before_feedback", sample_n=1)
        print_l2_norm_stats(v_new, name="v_profile_after_feedback", sample_n=1)
    
    if desparse:
        v_new = v_new.toarray()
    return v_new

def recommend_similar_to_article(article_id, X, articles_df, top_k=TOP_K_SIMILAR):
    """
    Recommande des articles similaires à celui identifié par article_id.
    
    Args:
        article_id (int): id de l'article de référence
        X (_array or sparse matrix_): matrice TF-IDF/Embed des articles
        articles_df (pd.DataFrame): catalogue des articles
        top_k (int, optional): nombre d'articles similaires à recommander. Defaults to TOP_K_SIMILAR.

    Raises:
        ValueError: si article_id inconnu dans le catalogue

    Returns:
        DataFrame: articles similaires recommandés
    """
    mask = (articles_df["id"] == article_id)
    if not mask.any():
        raise ValueError("article_id unknown in catalog")
    idx0 = np.where(mask.values)[0][0]
    # Ensure que X est 2D for sklearn
    X_mat = X
    if not issparse(X_mat):
        X_mat = np.asarray(X_mat)
        if X_mat.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features), got shape={X_mat.shape}")

    v = X_mat[idx0]
    # mtn si v est dense, on reshape en (1, D)
    if not issparse(v):
        v = np.asarray(v)
        if v.ndim == 1:
            v = v.reshape(1, -1)
    if os.getenv("DEBUG_L2_NORMS") == "1":
        print_l2_norm_stats(X, "X_sample", min(25, X.shape[0]))
        print_l2_norm_stats(v, "query_vec_from_catalog", 1)

    sims = linear_kernel(v, X).ravel()
    sims[idx0] = -np.inf
    idx = topk_indices(sims, top_k)
    return articles_df.iloc[idx]

def recommend_hot_articles(articles_df, method: str, top_k=TOP_K_MAIN, model=None, renew: bool = True):
    """
    method:
      - simple_count / weighted_count: utilise hot terms extraits des trend docs
      - tfidf: compare X_tfidf (articles) vs T_trends (trend docs transformes)
      - semantic: compare embeddings articles vs embeddings trend docs
    """
    df = articles_df.copy()
    if "text" not in df.columns:
        df["text"] = (df["title"].fillna("") + " " + df["abstract"].fillna("")).str.strip()

    trend_docs = get_hot_terms(df, top_n=10)
    trend_docs = [str(t) for t in trend_docs if t and str(t).strip()]

    if not trend_docs:
        df["trend_score"] = 0.0
    else:
        if method in ("simple_count", "weighted_count"):
            hot_terms = get_hot_term_list(trend_docs, top_n=HOT_TERMS_NUMB, renew_cache=renew)

            if method == "simple_count":
                df["trend_score"] = df["text"].str.lower().apply(lambda txt: sum(1 for t in hot_terms if t in txt)).astype(float)
            else:
                df["trend_score"] = df["text"].str.lower().apply(
                    lambda txt: float(sum(np.log1p(txt.count(t)) for t in hot_terms))
                )

        elif method == "tfidf":
            vectorizer, X = load_tfidf_elements()
            T = _cached_trend_tfidf_matrix(vectorizer, trend_docs)
            sims = linear_kernel(X, T)          # (N, n_trends)
            df["trend_score"] = mean_topk(sims, k=min(3, sims.shape[1]))

        elif method == "semantic":
            E, model2 = get_or_compute_article_embeddings(df, text_col="text")
            if model is None:
                model = model2
            Temb = encode_texts(trend_docs, model=model, normalize=True)
            sims = E @ Temb.T                    # (N, n_trends)
            df["trend_score"] = mean_topk(sims, k=min(3, sims.shape[1]))
        else:
            raise ValueError("method invalide: simple_count, weighted_count, tfidf or semantic")

    year_norm = safe_minmax_norm(df["year"].to_numpy()) if "year" in df.columns else 0.0
    cite_norm = safe_minmax_norm(df["cite_nb"].to_numpy()) if "cite_nb" in df.columns else 0.0

    df["final_hot_score"] = (
        0.5 * df["trend_score"].to_numpy(dtype=float) 
        + 0.3 * year_norm 
        + 0.2 * cite_norm
    )
    return df.sort_values("final_hot_score", ascending=False).head(top_k)

def recommend_for_query_text(query_text, transform, X, articles_df, top_k=TOP_K_MAIN):
    if transform not in ("tfidf", "embed"):
        raise ValueError("transform must be 'tfidf' or 'embed'")
    if transform == "embed":
        from src.embeddings import get_or_compute_article_embeddings, encode_texts
        _, model = get_or_compute_article_embeddings(articles_df, text_col="text")
        v_query = encode_texts([query_text], model=model, normalize=True)
    else:  # tfidf
        vectorizer, _ = load_tfidf_elements()
        v_query = vectorizer.transform([query_text])

    if os.getenv("DEBUG_L2_NORMS") == "1":
        from src.utils import print_l2_norm_stats
        print_l2_norm_stats(X, name="X_sample", sample_n=min(25, X.shape[0]))
        print_l2_norm_stats(v_query, name="v_query_from_text", sample_n=1)
    
    sims = linear_kernel(v_query, X).ravel()
    idx_sorted = sims.argsort()[::-1][:top_k]
    return articles_df.iloc[idx_sorted]