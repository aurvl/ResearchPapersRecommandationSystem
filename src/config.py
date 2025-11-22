from pathlib import Path

# Racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# API
ARXIV_API_URL = "https://export.arxiv.org/api/query"

# Dossiers de donnees
DATA_DIR = PROJECT_ROOT / "data"

# Cache
DATA_CACHE_DIR = DATA_DIR / "cache"

# Fichiers de donnees
ARTICLES_PATH = DATA_DIR / "articles_sample.csv"
PROFILE_KEYWORDS_PATH = DATA_DIR / "profile_keywords.csv"

# Dossier pour les modeles sauvegardes
MODELS_DIR = PROJECT_ROOT / "models"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"

# Parametres TF-IDF
TFIDF_MAX_FEATURES = 50000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_STOP_WORDS = "english"

# Parametres de recommandation
TOP_K_MAIN = 10          # nb d'articles proposés principalement
TOP_K_SIMILAR = 10       # nb d'articles similaires à proposer
PROFILE_ALPHA = 0.6     # 60% profil initial + 40% likes
RANDOM_SEED = 42
