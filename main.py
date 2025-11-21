# main.py
from src.data_loading import load_articles, load_profile_keywords
from src.text_vectorizer import fit_vectorizer
from src.profile_builder import build_profile_text, profile_to_vector
from src.recommender import (
    recommend_for_profile,
    recommend_hot_articles,
    recommend_similar_to_article,
    update_profile_with_likes,
)
from src.get_trends import get_hot_terms
from loguru import logger

def main():
    # 1) Charger les dat a
    logger.info("1. Loading data")
    print(">> Loading data...")
    articles_df = load_articles()
    profile_kw_df = load_profile_keywords()

    print(f"  - {len(articles_df)} articles loaded")
    print(articles_df[["id", "title", "field", "year"]].head(), "\n")

    # 2) TF-IDF sur le texte des articles
    logger.info("2. Fitting TF-IDF vectorizer")
    print(">> Fitting TF-IDF vectorizer...")
    vectorizer, X_tfidf = fit_vectorizer(articles_df["text"])
    print(f"  - Matrix shape: {X_tfidf.shape}\n")

    # 3) Simuler des préférences utilisateur
    #    (à remplacer par ton vrai formulaire plus tard)
    logger.info("3. Building user profile from simulated preferences")
    prefs = {
        "field": ["machine_learning", "recommender_systems"],
        "type": ["empirical"],
        "impact": ["high_impact"],
    }

    print(">> Building user profile from prefs:", prefs)
    profile_text = build_profile_text(prefs, profile_kw_df)
    print("  - Profile text:\n", profile_text[:300], "...\n")

    v_profile = profile_to_vector(profile_text, vectorizer)

    # 4) Recommandations basées sur le profil
    logger.info("4. Recommending articles for the user profile")
    print(">> Recommending articles for this profile...")
    recs_profile = recommend_for_profile(
        v_profile, X_tfidf, articles_df, top_k=5
    )
    print("Top-5 for profile:")
    print(recs_profile[["id", "title", "field", "year"]], "\n")

    # 5) Simuler des likes sur 2 premiers articles
    logger.info("5. Updating profile with user likes")
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
    
    # 6) Hot topics (arXiv ou corpus) + hot articles
    logger.info("6. Recommending hot articles based on trending topics")
    print(">> Getting hot terms (arXiv API or corpus fallback)...")
    hot_terms = get_hot_terms(articles_df, top_n=10)
    print("Hot terms:", hot_terms, "\n")

    print(">> Recommending hot articles...")
    recs_hot = recommend_hot_articles(articles_df, top_k=5)
    print("Top-5 hot articles:")
    print(recs_hot[["id", "title", "field", "year", "cite_nb"]], "\n")

    # 7) Articles similaires au premier article recommande pour le profil
    logger.info("7. Recommending articles similar to the first recommended article")
    first_id = recs_profile.iloc[0]["id"]
    print(f">> Recommending articles similar to {first_id} ...")
    recs_sim = recommend_similar_to_article(
        first_id, X_tfidf, articles_df, top_k=5
    )
    print("Similar articles:")
    print(recs_sim[["id", "title", "field", "year"]], "\n")

    print("Done.")


if __name__ == "__main__":
    main()
