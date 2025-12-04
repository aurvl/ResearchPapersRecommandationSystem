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
ARTICLES_PATH = DATA_DIR / "articles_sample.csv" # for tests (test avec petit dataset - partition du big df)
# ARTICLES_PATH = DATA_DIR / "articles.csv"
PROFILE_KEYWORDS_PATH = DATA_DIR / "profile_keywords.csv"

# Dossier pour les modeles sauvegardes
MODELS_DIR = PROJECT_ROOT / "models"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"

# Parametres TF-IDF
TFIDF_MAX_FEATURES = 500000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_STOP_WORDS = "english"

# Parametres de recommandation
TOP_K_MAIN    = 10          # nb d'articles recommandé principalement
TOP_K_SIMILAR = 10          # nb d'articles similaires à proposer
PROFILE_ALPHA = 0.6         # 60% profil initial + 40% likes
RANDOM_SEED   = 42

# Parametres de collecte OpenAlex
# Cible totale par catégorie
N_PER_CATEGORY = 750 

OUTPUT_FILE = DATA_DIR / "articles.csv"
OUTPUT_PARQUET = DATA_DIR / "articles.parquet" # pour format parquet

# Domaines
CATEGORY_LIST = [
    # --- COMPUTER SCIENCE & AI ---
    "Machine learning", "Deep learning", "Artificial intelligence", "Computer vision",
    "Natural language processing", "Robotics", "Reinforcement learning", "Neural network",
    "Generative adversarial network", "Transformer (machine learning function)", "Large language model",
    "Recommender system", "Information retrieval", "Data mining", "Big data",
    "Human–computer interaction", "Cybersecurity", "Cryptography", "Blockchain",
    "Cloud computing", "Edge computing", "Internet of things", "Distributed computing",
    "Software engineering", "Operating system", "Database", "Computer network",
    "Quantum computing", "Bioinformatics", "Computational biology",

    # --- ECONOMICS, FINANCE & BUSINESS ---
    "Econometrics", "Macroeconomics", "Microeconomics", "Development economics",
    "Behavioral economics", "Labor economics", "International trade",
    "Finance", "Financial market", "Asset pricing", "Portfolio optimization",
    "Risk management", "Corporate finance", "Financial technology", "Cryptocurrency",
    "Marketing", "Supply chain management", "Operations research", "Game theory",

    # --- MATHEMATICS & STATISTICS ---
    "Statistics", "Probability theory", "Bayesian inference", "Time series",
    "Mathematical optimization", "Linear algebra", "Topology", "Differential equation",
    "Combinatorics", "Number theory", "Geometry", "Graph theory", "Stochastic process",

    # --- PHYSICS & ASTRONOMY ---
    "Physics", "Quantum mechanics", "Particle physics", "Astrophysics", "Cosmology",
    "General relativity", "Condensed matter physics", "Nanotechnology", "Optics",
    "Thermodynamics", "Fluid dynamics", "Plasma physics", "Nuclear physics",
    "Geophysics", "Biophysics", "Materials science",

    # --- CHEMISTRY & MATERIALS ---
    "Chemistry", "Organic chemistry", "Inorganic chemistry", "Biochemistry",
    "Physical chemistry", "Polymer", "Nanomaterials", "Chemical engineering",
    "Metallurgy", "Crystallography",

    # --- BIOLOGY & MEDICINE ---
    "Biology", "Genetics", "Genomics", "Molecular biology", "Cell biology",
    "Neuroscience", "Cognitive science", "Immunology", "Virology", "Microbiology",
    "Epidemiology", "Public health", "Cancer research", "Medicine", "Pharmacology",
    "Biotechnology", "CRISPR", "Synthetic biology",

    # --- EARTH & ENVIRONMENT ---
    "Environmental science", "Climate change", "Global warming", "Ecology",
    "Biodiversity", "Oceanography", "Geology", "Atmospheric science",
    "Renewable energy", "Solar energy", "Wind power", "Sustainability",
    "Hydrology", "Agriculture", "Agronomy",

    # --- ENGINEERING ---
    "Engineering", "Electrical engineering", "Mechanical engineering", "Civil engineering",
    "Aerospace engineering", "Robotics", "Control theory", "Signal processing",
    "Image processing", "Telecommunications", "Electronics", "3D printing",

    # --- SOCIAL SCIENCES ---
    "Psychology", "Social psychology", "Sociology", "Political science",
    "International relations", "Education", "Pedagogy", "Linguistics",
    "Law", "Philosophy", "Urban planning", "Smart city"
]
