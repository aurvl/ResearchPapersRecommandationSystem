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

def profile_to_vector(profile_text: str, vectorizer):
    return vectorizer.transform([profile_text])
