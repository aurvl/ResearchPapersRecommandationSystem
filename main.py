# main.py
import joblib
import numpy as np
from src.data_loading import load_articles, load_profile_keywords
from src.text_vectorizer import fit_vectorizer
from src.profile_builder import build_profile_text, profile_from_text, profile_to_vector
from src.recommender import (
    recommend_for_profile,
    recommend_hot_articles,
    recommend_similar_to_article,
    update_profile_with_likes,
    # Embeddings versions
    recommend_for_profile_emb,
    update_profile_with_likes_emb,
    recommend_similar_to_article_emb,
)
from src.embeddings import get_embedding_model, encode_texts, get_or_compute_article_embeddings
from src.get_trends import get_hot_terms
from loguru import logger
from src.config import TFIDF_VECTORIZER_PATH, HF_EMBED_MODEL_NAME, EMBEDDINGS_CACHE_PATH

# ========== REPRESENTATION MODE ==========
# Options: "tfidf" | "embeddings" | "both"
REPRESENTATION_MODE = "both"
# =========================================


def main():
    # 1) Charger les données
    logger.info("1. Loading data")
    print(">> Loading data...")
    articles_df = load_articles()
    profile_kw_df = load_profile_keywords()

    print(f"  - {len(articles_df)} articles loaded")
    print(articles_df[["id", "title", "field", "year"]].head(), "\n")

    # 2) TF-IDF sur le texte des articles (toujours chargé, même si on n'utilise que embeddings)
    logger.info("2. Fitting TF-IDF vectorizer")
    print(">> Fitting TF-IDF vectorizer...")
    vectorizer, X_tfidf = fit_vectorizer(articles_df["text"])
    print(f"  - Matrix shape: {X_tfidf.shape}\n")

    # 2b) Charger ou calculer les embeddings si nécessaire
    article_embeddings = None
    embedding_model = None
    
    if REPRESENTATION_MODE in ["embeddings", "both"]:
        try:
            logger.info("2b. Loading/Computing article embeddings")
            print(">> Loading/Computing article embeddings (MiniLM)...")
            article_embeddings = get_or_compute_article_embeddings(
                articles_df,
                text_col="text",
                cache_path=str(EMBEDDINGS_CACHE_PATH),
                model_name=HF_EMBED_MODEL_NAME
            )
            print(f"  - Embeddings shape: {article_embeddings.shape}\n")
            
            # Charger le modèle pour encoder le profil utilisateur
            embedding_model = get_embedding_model(HF_EMBED_MODEL_NAME)
            
        except Exception as e:
            logger.warning(f"Failed to load embeddings: {e}. Falling back to TF-IDF only.")
            print(f"⚠️  Warning: Embeddings failed ({e}). Using TF-IDF only.\n")
            article_embeddings = None
            embedding_model = None
            if REPRESENTATION_MODE == "embeddings":
                REPRESENTATION_MODE = "tfidf"
            elif REPRESENTATION_MODE == "both":
                pass  # Continue with TF-IDF only

    # 3) Simuler des préférences utilisateur
    #    (à remplacer par ton vrai formulaire plus tard)
    logger.info("3. Building user profile")
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
    print("  - Profile text:\n", profile_text[:300], "...\n")

    v_profile = profile_to_vector(profile_text, vectorizer)
    v_profile_enriched = profile_to_vector(enriched_text, vectorizer)
    
    # Préparer embeddings du profil si nécessaire
    profile_embedding = None
    profile_embedding_enriched = None
    if article_embeddings is not None and embedding_model is not None:
        profile_embedding = encode_texts([profile_text], embedding_model, normalize=True)
        profile_embedding_enriched = encode_texts([enriched_text], embedding_model, normalize=True)

    # 4) Recommandations basées sur le profil
    
    # ========== TF-IDF RECOMMENDATIONS ==========
    if REPRESENTATION_MODE in ["tfidf", "both"]:
        print("\n" + "="*60)
        print("===           TF-IDF RESULTS                           ===")
        print("="*60 + "\n")
        
        logger.info("4.a. Recommending articles for the user profile (TF-IDF)")
        print(">> Recommending articles for this profile (TF-IDF)...")
        recs_profile = recommend_for_profile(
            v_profile, X_tfidf, articles_df, top_k=5
        )
        print("Top-5 for profile:")
        print(recs_profile[["id", "title", "field", "year"]], "\n")
        
        logger.info("4.b. Recommending articles for the enriched text profile (TF-IDF)")
        print(">> Recommending articles for the enriched text profile (TF-IDF)...")
        recs_enriched = recommend_for_profile(
            v_profile_enriched, X_tfidf, articles_df, top_k=5
        )
        print("Top-5 for enriched text profile:")
        print(recs_enriched[["id", "title", "field", "year"]], "\n")

        # 5) Simuler des likes sur 2 premiers articles
        logger.info("5. Updating profile with user likes (TF-IDF)")
        liked_ids = recs_profile["id"].iloc[:2].tolist()
        print("User likes these articles:", liked_ids)

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
        
        # 7) Articles similaires au premier article recommande pour le profil
        logger.info("7. Recommending articles similar to the first recommended article (TF-IDF)")
        first_id = recs_profile.iloc[0]["id"]
        print(f">> Recommending articles similar to {first_id} (TF-IDF)...")
        recs_sim = recommend_similar_to_article(
            first_id, X_tfidf, articles_df, top_k=5
        )
        print("Similar articles:")
        print(recs_sim[["id", "title", "field", "year"]], "\n")
    
    # ========== EMBEDDINGS RECOMMENDATIONS ==========
    if REPRESENTATION_MODE in ["embeddings", "both"] and article_embeddings is not None:
        print("\n" + "="*60)
        print("===      EMBEDDINGS (MiniLM) RESULTS                   ===")
        print("="*60 + "\n")
        
        logger.info("4. Recommending articles for the user profile (Embeddings)")
        print(">> Recommending articles for this profile (Embeddings)...")
        recs_profile_emb = recommend_for_profile_emb(
            profile_embedding, article_embeddings, articles_df, top_k=5
        )
        print("Top-5 for profile (embeddings):")
        print(recs_profile_emb[["id", "title", "field", "year"]], "\n")

        # 5) Simuler des likes sur 2 premiers articles
        logger.info("5. Updating profile with user likes (Embeddings)")
        liked_ids_emb = recs_profile_emb["id"].iloc[:2].tolist()
        print("User likes these articles:", liked_ids_emb)

        profile_embedding_updated = update_profile_with_likes_emb(
            profile_embedding, liked_ids_emb, article_embeddings, articles_df
        )

        recs_after_likes_emb = recommend_for_profile_emb(
            profile_embedding_updated,
            article_embeddings,
            articles_df,
            top_k=5,
            exclude_ids=set(liked_ids_emb),
        )

        print("Top-5 after likes (embeddings):")
        print(recs_after_likes_emb[["id", "title", "field", "year"]], "\n")
        
        # 7) Articles similaires au premier article recommande pour le profil
        logger.info("7. Recommending articles similar to the first recommended article (Embeddings)")
        first_id_emb = recs_profile_emb.iloc[0]["id"]
        print(f">> Recommending articles similar to {first_id_emb} (Embeddings)...")
        recs_sim_emb = recommend_similar_to_article_emb(
            first_id_emb, article_embeddings, articles_df, top_k=5
        )
        print("Similar articles (embeddings):")
        print(recs_sim_emb[["id", "title", "field", "year"]], "\n")

    # 6) Hot topics (arXiv ou corpus) + hot articles (mode agnostique)
    print("\n" + "="*60)
    print("===           HOT ARTICLES (Mode-Agnostic)             ===")
    print("="*60 + "\n")
    
    logger.info("6. Recommending hot articles based on trending topics")
    print(">> Getting hot terms (arXiv API or corpus fallback)...")
    hot_terms = get_hot_terms(articles_df, top_n=10)
    print("Hot terms:", hot_terms, "\n")

    print(">> Recommending hot articles...")
    recs_hot = recommend_hot_articles(articles_df, top_k=5)
    print("Top-5 hot articles:")
    print(recs_hot[["id", "title", "field", "year", "cite_nb"]], "\n")

    # Sauvegarde du vectorizer TF-IDF
    joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
