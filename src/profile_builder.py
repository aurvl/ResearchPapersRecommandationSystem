import pandas as pd

def build_profile_text(preferences: dict, profile_kw_df: pd.DataFrame) -> str:
    """
    preferences: dict comme {"field": ["machine_learning", "finance"],
                              "type": ["empirical"]}
    profile_kw_df: colonnes [dimension, option, keywords]
    """
    tokens = []
    for dim, options in preferences.items():
        if not isinstance(options, list):
            options = [options]
        for opt in options:
            mask = (profile_kw_df["dimension"] == dim) & (profile_kw_df["option"] == opt)
            matches = profile_kw_df.loc[mask]
            if not matches.empty:
                tokens.append(matches.iloc[0]["keywords"])
    return " ".join(tokens)

def profile_from_text(raw_text: str, profile_kw_df: pd.DataFrame) -> str:
    """
    Prend un texte brut (ex: "Attention is all you need"), le nettoie,
    et l'enrichit avec les mots-clés du CSV si des concepts sont détectés.
    """
    if not raw_text:
        return ""
    
    text_lower = raw_text.lower().strip()
    enrichments = []
    
    input_tokens = set(text_lower.split())
    for _, row in profile_kw_df.iterrows():
        # Options nettoyées (ex: "deep_learning" => "deep learning")
        option_clean = row['option'].replace("_", " ")
        keywords_str = row['keywords']
        
        # logic
        # A. Est-ce que le nom d'une catégorie est dans le text ? On parcours row by row pour trouver
        if option_clean in text_lower:
            enrichments.append(keywords_str)
            continue
        # B. Est-ce que des mots-clés spécifiques sont dans le texte ?
        kw_tokens = set(keywords_str.split())
        # Intersection : Si on a des mots en commun entre l'entrée et la ligne du CSV
        if not input_tokens.isdisjoint(kw_tokens):
            enrichments.append(keywords_str)

    full_text = text_lower + " " + " ".join(enrichments)
    return full_text

def profile_to_vector(profile_text: str, vectorizer):
    return vectorizer.transform([profile_text])
