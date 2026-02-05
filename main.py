# main.py
from src.data_loading import load_articles, load_profile_keywords, query_to_text
from src.text_vectorizer import fit_vectorizer, save_tfidf_elements, load_tfidf_elements
from src.profile_builder import build_profile_text, profile_to_vector, profile_from_text
from src.recommender import (
    recommend_for_profile,
    recommend_hot_articles,
    recommend_similar_to_article,
    update_profile_with_likes,
    recommend_for_query_text,
)
from src.embeddings import get_or_compute_article_embeddings, encode_texts
from src.get_trends import get_hot_terms, get_hot_term_list
from loguru import logger
from src.config import TFIDF_VECTORIZER_PATH, ARTICLES_PATH

def main():
    # I. Charger les données
    logger.info("I. Loading data")
    print(">> Loading data...")
    articles_df = load_articles(ARTICLES_PATH)
    profile_kw_df = load_profile_keywords()

    print(f"  - {len(articles_df)} articles loaded")
    print(articles_df[["id", "title", "field", "year"]].head(), "\n")

    # II. TF-IDF sur le texte des articles
    logger.info("II. TF-IDF vectorization")
    logger.info("1. Fitting TF-IDF vectorizer")
    print(">> Fitting TF-IDF vectorizer...")
    vectorizer, X_tfidf = fit_vectorizer(articles_df["text"]) # fitting
    # vectorizer, X_tfidf = load_tfidf_elements() # loading cached
    # print(f"  - Vectorizer loaded from {TFIDF_VECTORIZER_PATH}")
    print(f"  - Matrix shape: {X_tfidf.shape}\n")

    # 1) Simuler des préférences utilisateur
    #    (à remplacer par ton vrai formulaire plus tard)
    logger.info("2. Building user profile")
    prefs = {
        "field": ["machine_learning", "recommender_systems"],
        "type": ["empirical"],
        "impact": ["high_impact"],
    }
    
    print(">> a. Build user profile from user inputs...")
    user_input = "Attention is all you need"
    print(f">> Input brut: {user_input}")
    enriched_text = profile_from_text(user_input, profile_kw_df)
    print(f">> Texte enrichi: {enriched_text[:200]}...") 

    print('\n')
    print(">> b. Building user profile from prefs:", prefs)
    profile_text = build_profile_text(prefs, profile_kw_df)
    print("  - Profile text:\n", profile_text[:200], "...\n")

    v_profile = profile_to_vector(profile_text, vectorizer)
    v_profile_enriched = profile_to_vector(enriched_text, vectorizer)

    # 2) Recommandations basées sur le profil
    logger.info("3.a. Recommending articles for the user profile")
    print(">> Recommending articles for this profile...")
    recs_profile = recommend_for_profile(
        v_profile, X_tfidf, articles_df, top_k=5
    )
    print("Top-5 for profile:")
    print(recs_profile[["id", "title", "field", "year"]], "\n")
    
    logger.info("3.b. Recommending articles for the enriched text profile")
    print(">> Recommending articles for the enriched text profile...")
    recs_enriched = recommend_for_profile(
        v_profile_enriched, X_tfidf, articles_df, top_k=5
    )
    print("Top-5 for enriched text profile:")
    print(recs_enriched[["id", "title", "field", "year"]], "\n")

    # 3) Simuler des likes sur 2 premiers articles
    logger.info("4. Updating profile with user likes")
    print("the user likes the first 2 recommended articles...")
    liked_ids = recs_profile["id"].iloc[:2].tolist()
    print("He likes these articles:", liked_ids)

    v_profile_updated = update_profile_with_likes(
        v_profile, liked_ids, X_tfidf, articles_df
    )

    recs_after_likes = recommend_for_profile(
        v_profile_updated,
        X_tfidf,
        articles_df,
        top_k=5,
        exclude_ids=set(liked_ids),  # pour ne pas reproposer les memes
    )

    print("Top-5 after likes (updated profile):")
    print(recs_after_likes[["id", "title", "field", "year"]], "\n")
    
    # 3) Hot topics (arXiv ou corpus) + hot articles
    logger.info("5. Recommending hot articles based on trending topics")
    print(">> Getting hot terms (arXiv API or corpus fallback)...")
    hot_terms = get_hot_terms(articles_df, top_n=10)
    print("Hot articles (title + abstract):\n")
    for i, t in enumerate(hot_terms, 1):
        print(f"{i:02d}. {t}")
    
    trend_docs = [str(x) for x in hot_terms]
    terms_hots = get_hot_term_list(trend_docs, renew_cache=False)
    top_tens = terms_hots[:10]
    print("\nHOT TERMS (from trend_docs):")
    for i, t in enumerate(top_tens, 1):
        print(f"{i:02d}. {t}")

    print(">> Recommending hot articles...")
    recs_hot = recommend_hot_articles(articles_df, method="tfidf", top_k=5)
    print("Top-5 hot articles:")
    print(recs_hot[["id", "title", "field", "year", "cite_nb"]], "\n")

    # 4) Articles similaires au premier article recommandé pour le profil
    logger.info("6. Recommending articles similar to the first recommended article")
    first_id = recs_profile.iloc[0]["id"]
    print(f">> Recommending articles similar to {first_id} ...")
    recs_sim = recommend_similar_to_article(
        first_id, X_tfidf, articles_df, top_k=5
    )
    print("Similar articles:")
    print(recs_sim[["id", "title", "field", "year"]], "\n")

    # 5) Recommandation contextuelle avancée (article externe)
    logger.info("7. Recommending articles similar to an external article")
    print(">> Loading external article from payload.json...")
    query = query_to_text("data/payload.json")
    print(">> Recommending articles similar to the external article...")
    recs_external = recommend_for_query_text(
        query, 'tfidf', X_tfidf, articles_df, top_k=5
    )
    print("Similar articles to external article:")
    print(recs_external[["id", "title", "field", "year"]])
    
    # III. Embeddings-based recommendation for external article
    # 1) Compute/load article embeddings and build profile embedding
    logger.info("III. Embeddings-based recommendation for external article")
    logger.info("1. Computing/loading article embeddings")
    X_emb, model = get_or_compute_article_embeddings(articles_df)
    print(f"  - Embedding matrix shape: {X_emb.shape}\n")
    
    logger.info("2. Recomputing profile vectors with embeddings...")
    print(">> Building user profile from prefs:", prefs)
    if X_emb is not None and model is not None:
        profile_embedding = encode_texts([profile_text], model, normalize=True)
    print("  - Profile embedding:", profile_embedding.shape)
    
    # 2) Recommandations basées sur le profil avec embeddings
    logger.info("3. Recommending articles for the user profile with embeddings")
    print(">> Recommending articles for this profile with embeddings...")
    recs_profile = recommend_for_profile(
        profile_embedding, X_emb, articles_df, top_k=5
    )
    print("Top-5 for profile:")
    print(recs_profile[["id", "title", "field", "year"]], "\n")
    
    logger.info("4. Updating profile with user likes using embeddings")
    print(">> Recommending articles based on likes with embeddings...")
    print("The user likes again the first 2 recommended articles...")
    profile_emb_updated = update_profile_with_likes(
        profile_embedding, liked_ids, X_emb, articles_df, desparse=True
    )
    print("Updated profile vector:\n", profile_emb_updated.shape)
    
    recs_after_likes = recommend_for_profile(
        profile_emb_updated,
        X_emb,
        articles_df,
        top_k=5,
        exclude_ids=set(liked_ids),  # pour ne pas reproposer les memes
    )

    print(">> Top-5 after likes (updated profile):")
    print(recs_after_likes[["id", "title", "field", "year"]], "\n")
    
    # 3) Hot topics + hot articles avec embeddings
    logger.info("5. Recommending hot articles based on trending topics with embeddings")
    print(">> Recommending hot articles...")
    recs_hot = recommend_hot_articles(articles_df, method="semantic", model=model, top_k=5)
    print("Top-5 hot articles:")
    print(recs_hot[["id", "title", "field", "year", "cite_nb"]], "\n")
    
    # 4) Articles similaires au premier article recommandé pour le profil avec embeddings
    logger.info("6. Recommending articles similar to the first recommended article with embeddings")
    first_id = recs_profile.iloc[0]["id"]
    print(f">> Recommending articles similar to {first_id} ...")
    recs_sim = recommend_similar_to_article(
        first_id, X_emb, articles_df, top_k=5
    )
    print("Similar articles:")
    print(recs_sim[["id", "title", "field", "year"]], "\n")
    
    logger.info("7. Recommending articles similar to an external article with embeddings")
    print(">> Recommending articles similar to the external article...")
    recs_external = recommend_for_query_text(
        query, 'embed', X_emb, articles_df, top_k=5
    )
    print("Similar articles to external article:")
    print(recs_external[["id", "title", "field", "year"]], "\n")
    
    print("Done !")


if __name__ == "__main__":
    main()
