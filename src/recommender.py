import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics.pairwise import linear_kernel
from src.config import TOP_K_MAIN, TOP_K_SIMILAR, PROFILE_ALPHA
from src.get_trends import get_hot_terms

def recommend_for_profile(v_profile, X_tfidf, articles_df, top_k=TOP_K_MAIN, exclude_ids=None):
    if exclude_ids is None:
        exclude_ids = set()
    sims = linear_kernel(v_profile, X_tfidf).ravel()
    # on exclut certains articles si besoin
    mask = ~articles_df["id"].isin(exclude_ids)
    sims_filtered = sims.copy()
    sims_filtered[~mask.values] = -1  # penalite
    idx_sorted = sims_filtered.argsort()[::-1][:top_k]
    return articles_df.iloc[idx_sorted]
    
def update_profile_with_likes(v_profile, liked_ids, X_tfidf, articles_df, alpha=PROFILE_ALPHA):
    """
    v_profile : vecteur TF-IDF (1, D) du profil courant
    liked_ids : liste D id d'articles que l'utilisateur a likés
    X_tfidf   : matrice TF-IDF des articles
    articles_df : DataFrame des articles
    alpha     : poids du profil initial vs likes (0 <= alpha <= 1)

    Retourne un nouveau vecteur de profil v_new.
    """
    if not liked_ids:
        return v_profile

    mask = articles_df["id"].isin(liked_ids)
    if not mask.any():
        return v_profile

    liked_vecs = X_tfidf[mask.values]         # vecteurs des articles likés -> sparse (n_liked, D)
    liked_centroid = liked_vecs.mean(axis=0)  # centroide (1, D)

    # assurer CSR : debug
    if not issparse(liked_centroid):
        liked_centroid = csr_matrix(liked_centroid)

    if not issparse(v_profile):
        v_profile = csr_matrix(v_profile)
    
    v_new = alpha * v_profile + (1 - alpha) * liked_centroid
    return v_new

def recommend_similar_to_article(article_id, X_tfidf, articles_df, top_k=TOP_K_SIMILAR):
    mask = (articles_df["id"] == article_id)
    if not mask.any():
        raise ValueError("article_id inconnu")
    idx = np.where(mask.values)[0][0]
    vec = X_tfidf[idx]
    sims = linear_kernel(vec, X_tfidf).ravel()
    sims[idx] = -1  # on exclut l'article lui-meme
    idx_sorted = sims.argsort()[::-1][:top_k]
    return articles_df.iloc[idx_sorted]


def recommend_hot_articles(articles_df, top_k=TOP_K_MAIN):
    articles_df = articles_df.copy()

    # 1) Récupère les tendances (arXiv ou fallback interne)
    trends = get_hot_terms(articles_df, top_n=10)
    trends = [t.lower() for t in trends]

    if "text" not in articles_df.columns:
        articles_df["text"] = articles_df["title"] + " " + articles_df["abstract"]

    # 2) trend_score
    def compute_trend_score(text: str):
        txt = text.lower()
        return sum(t in txt for t in trends)

    articles_df["trend_score"] = articles_df["text"].apply(compute_trend_score)

    # 3) normalisation recence + citations
    year_norm = (articles_df["year"] - articles_df["year"].min()) / \
                (articles_df["year"].max() - articles_df["year"].min())

    cite_norm = (articles_df["cite_nb"] - articles_df["cite_nb"].min()) / \
                (articles_df["cite_nb"].max() - articles_df["cite_nb"].min())

    # 4) Score final
    articles_df["final_hot_score"] = (
        0.5 * articles_df["trend_score"]
        + 0.3 * year_norm
        + 0.2 * cite_norm
    )

    return articles_df.sort_values("final_hot_score", ascending=False).head(top_k)
