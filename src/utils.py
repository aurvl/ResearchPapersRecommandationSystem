import requests
import time
import math
from datetime import datetime
import numpy as np
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.preprocessing import normalize

# email requested by OpenAlex
MY_EMAIL = "aurelvvince@gmail.com"

def reconstruct_abstract(inverted_index):
    """
    Reconstruit le texte lisible d'un abstract à partir de l'index inversé d'OpenAlex.

    OpenAlex ne stock pas le texte brut de l'abstratc pour des raisons de droits d'auteur (contournement 
    des restrictions des éditeurs) et de compression mais fournit un dict où chaque 
    mot est associé à sa position dans la phrase.
    par ex :
        Entrée : {"L'IA": [0], "futur": [3], "le": [2], "est": [1]}
        Sortie : "L'IA est le futur"

    Args:
        inverted_index (dict): Dictionnaire {mot: [position]}.

    Returns:
        str: Le texte reconstruit dans l'ordre chronologique.
    """
    if not inverted_index:
        return ""
    word_list = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_list.append((pos, word))
    sorted_words = sorted(word_list, key=lambda x: x[0])
    return " ".join([w[1] for w in sorted_words])


def get_concept_id(concept_name):
    """Récupère l'ID OpenAlex d'un concept (ex: Machine Learning -> C154945302)"""
    url = "https://api.openalex.org/concepts"
    params = {"search": concept_name, "per_page": 1}
    headers = {"User-Agent": f"mailto:{MY_EMAIL}"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if r.status_code == 200:
            results = r.json().get('results', [])
            if results:
                return results[0]['id']
    except Exception as e:
        print(f"Error finding concept '{concept_name}': {e}")
    return None


# ---------- IMAGE MAPPING ----------

FIELD_TO_IMAGE_MAP = {
    "computer_science_and_ai.png": [
        "machine_learning", "deep_learning", "artificial_intelligence", "computer_vision", "nlp", "robotics", 
        "neural_network", "gan", "transformer", "llm", "recommender_system", "data_mining", "big_data", "hci", 
        "cybersecurity", "blockchain", "cloud_computing", "iot", "software_engineering", "database"
    ],
    "economics_finance_and_business.png": [
        "econometrics", "macroeconomics", "microeconomics", "finance", "asset_pricing", "portfolio_optimization", 
        "risk_management", "corporate_finance", "fintech", "crypto", "marketing", "supply_chain", "game_theory"
    ],
    "mathematics_and_statistics.png": [
        "statistics", "probability", "bayesian_inference", "time_series", "optimization", "linear_algebra", 
        "topology", "differential_equation", "combinatorics", "number_theory", "graph_theory"
    ],
    "physics_and_astronomy.png": [
        "physics", "quantum_mechanics", "astrophysics", "cosmology", "relativity", "nanotechnology", "optics", 
        "fluid_dynamics", "nuclear_physics"
    ],
    "chemistry_and_materials.png": [
        "chemistry", "organic_chemistry", "inorganic_chemistry", "biochemistry", "polymer", "nanomaterials", 
        "chemical_engineering", "crystallography"
    ],
    "biology_and_medicine.png": [
        "biology", "genetics", "genomics", "neuroscience", "cognitive_science", "immunology", "virology", 
        "epidemiology", "public_health", "medicine", "pharmacology", "crispr"
    ],
    "earth_and_environment.png": [
        "environmental_science", "climate_change", "ecology", "biodiversity", "oceanography", "geology", 
        "renewable_energy", "solar_power", "wind_power", "sustainability", "agriculture"
    ],
    "engineering.png": [
        "engineering", "electrical", "mechanical", "civil", "aerospace", "control_theory", "signal_processing", 
        "telecommunications", "electronics", "3d_printing"
    ],
    "social_sciences.png": [
        "psychology", "sociology", "political_science", "international_relations", "education", "law", 
        "philosophy", "urban_planning", "smart_city"
    ]
}

# Invert the map for O(1) lookup: { "machine_learning": "computer_science_and_ai.png", ... }
FIELD_LOOKUP = {}
for img, fields in FIELD_TO_IMAGE_MAP.items():
    for f in fields:
        # Normalize keys: lowercase and replace spaces with underscores just in case
        key = f.lower().replace(" ", "_")
        FIELD_LOOKUP[key] = img

def get_article_image(field_name):
    """
    Returns the static image path based on the article's field.
    """
    if not field_name:
        return "/static/img/computer_science_and_ai.png" # Default fallback
    
    # Normalize input field
    normalized_field = str(field_name).lower().strip().replace(" ", "_")
    
    # Lookup
    image_filename = FIELD_LOOKUP.get(normalized_field)
    
    if image_filename:
        return f"/static/img/{image_filename}"
    
    # Fallback if field not found
    return "/static/img/computer_science_and_ai.png"


def fetch_papers_by_concept(concept_id, concept_name, total_limit):
    """
    Récupère les articles de manière représentative (Uniforme sur les années).
    Stratégie : Loop 2010 -> 2025 et prend ~X articles par an.
    """
    works_url = "https://api.openalex.org/works" # url
    collected = []
    
    # Définition de la plage temporelle de 2010 a ajd
    start_year = 2010
    current_year = datetime.now().year + 1 # Inclure 2025/2026
    years = list(range(start_year, current_year))
    
    # Quota de papiers a prendre par année (750 articles / 16 ans (2010-2025) = 46 articles/an)
    limit_per_year = math.ceil(total_limit / len(years))

    select_fields = (
        "id,title,abstract_inverted_index,publication_year,"
        "primary_location,authorships,cited_by_count,doi,ids"
    )

    print(f"   -> Harvesting '{concept_name}' (~{limit_per_year} papers/year)...")

    for year in years:
        # Bloqué si limit atteinte (pour sécu la collect)
        if len(collected) >= total_limit:
            break

        params = {
            "filter": f"concepts.id:{concept_id},publication_year:{year}",
            "select": select_fields,
            "per_page": limit_per_year, # On prend exactement le quota requis sur l'année
            # most cited par an
            "sort": "cited_by_count:desc" 
        }
        headers = {"User-Agent": f"mailto:{MY_EMAIL}"}

        try:
            # requete
            r = requests.get(works_url, params=params, headers=headers, timeout=10)
            
            if r.status_code != 200:
                print(f"      Error year {year}: {r.status_code}") # msg d'erreur si ca fail
                continue
            
            data = r.json()
            results = data.get('results', [])
            
            if not results:
                continue

            for work in results:
                # Vérification qualité data (Titre + Abstract obligatoires)
                if not work.get('title') or not work.get('abstract_inverted_index'):
                    continue
                
                # Extract auteurs names
                authors = []
                for auth in work.get('authorships', []):
                    if auth.get('author'):
                        authors.append(auth['author'].get('display_name', ''))
                author_str = ", ".join(authors)

                # Extract Journal
                journal = "Unknown"
                loc = work.get('primary_location') or {}
                if loc.get('source'):
                    journal = loc['source'].get('display_name', 'Unknown')

                # Extract URL
                url = work.get('doi') or work.get('ids', {}).get('openalex', '')

                # dcp stocker tout dans la list
                collected.append({
                    "id": work['id'].replace("https://openalex.org/", ""),
                    "title": work['title'],
                    "abstract": reconstruct_abstract(work['abstract_inverted_index']),
                    "year": work['publication_year'],
                    "field": concept_name,
                    "url": url,
                    "author": author_str,
                    "journal": journal,
                    "cite_nb": work['cited_by_count']
                })

            # Petite pause pour l'API (eviter de se faire eject)
            time.sleep(0.1)

        except Exception as e:
            print(f"      Crash year {year}: {e}")
            continue
            
    return collected


def print_l2_norm_stats(X, name: str = "X", sample_n: int = 10):
    """Print L2 norm stats (min/max/mean) for a sparse matrix/vector.

    Designed for TF-IDF sanity checks without densifying.
    - For 2D matrices: computes row-wise L2 norms.
    - For 1D vectors: treats as a single row.
    """
    if X is None:
        print(f"[norms] {name}: None")
        return

    if issparse(X):
        n_rows = X.shape[0]
        if n_rows == 0:
            print(f"[norms] {name}: empty")
            return
        k = max(1, min(int(sample_n), n_rows))
        Xs = X[:k]
        norms_sq = Xs.multiply(Xs).sum(axis=1)
        norms = np.sqrt(np.asarray(norms_sq).ravel())
    else:
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        n_rows = X_arr.shape[0]
        if n_rows == 0:
            print(f"[norms] {name}: empty")
            return
        k = max(1, min(int(sample_n), n_rows))
        Xs = X_arr[:k]
        norms = np.linalg.norm(Xs, axis=1)

    finite = norms[np.isfinite(norms)]
    if finite.size == 0:
        print(f"[norms] {name}: no finite norms")
        return

    zero_count = int(np.sum(finite == 0.0))
    print(
        f"[norms] {name}: rows_sampled={int(len(norms))} "
        f"min={finite.min():.6f} max={finite.max():.6f} mean={finite.mean():.6f} zeros={zero_count}"
    )

def safe_minmax_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn)

def mean_topk(scores_2d: np.ndarray, k: int = 3) -> np.ndarray:
    if scores_2d.size == 0:
        return np.zeros((scores_2d.shape[0],), dtype=float)
    k = max(1, min(k, scores_2d.shape[1]))
    part = np.partition(scores_2d, -k, axis=1)[:, -k:]
    return part.mean(axis=1)

def topk_indices(scores: np.ndarray, k: int, exclude_mask: np.ndarray | None = None) -> np.ndarray:
    """Retourne indices top-k (desc). Optionnel: mask a exclure (True = exclu)."""
    s = np.asarray(scores, dtype=float)
    if exclude_mask is not None:
        s = s.copy()
        s[exclude_mask] = -np.inf

    k = max(1, min(int(k), s.size))
    idx = np.argpartition(s, -k)[-k:]
    idx = idx[np.argsort(s[idx])[::-1]]
    return idx


def _is_sparse_mode(X, mode=None):
    if mode is None:
        return sp.issparse(X)
    mode = str(mode).lower().strip()
    if mode not in {"sparse", "dense"}:
        raise ValueError('mode must be "sparse" or "dense" (or None)')
    return mode == "sparse"


def _prep_X(X, *, is_sparse: bool, preproc=None):
    """Prepare X for reducer.transform with consistent preprocessing."""
    if is_sparse:
        if not sp.issparse(X):
            raise TypeError("Expected a sparse matrix for mode='sparse'")
        Xp = X.tocsr().astype(np.float32)
        Xp = normalize(Xp, norm="l2", axis=1, copy=False)
        return Xp
    else:
        Xp = np.asarray(X, dtype=np.float32)
        if Xp.ndim == 1:
            Xp = Xp.reshape(1, -1)
        if preproc is not None:
            Xp = preproc.transform(Xp)
        return Xp


def _prep_v(v, *, is_sparse: bool, preproc=None):
    """Prepare a single vector (profile) for reducer.transform."""
    if v is None:
        return None
    if is_sparse:
        if sp.issparse(v):
            vr = v.tocsr()
            if vr.shape[0] != 1:
                vr = vr.reshape(1, -1)
        else:
            arr = np.asarray(v)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            vr = sp.csr_matrix(arr)
        vr = vr.astype(np.float32)
        vr = normalize(vr, norm="l2", axis=1, copy=False)
        return vr
    else:
        vr = np.asarray(v, dtype=np.float32)
        if vr.ndim == 1:
            vr = vr.reshape(1, -1)
        if preproc is not None:
            vr = preproc.transform(vr)
        return vr


def plot_article_space(
    X, reducer, recs_df, *,
    v_profile=None,
    pc_x=1, pc_y=2,
    sample_bg=8000,
    annotate_top=False,
    random_state=42,
    mode=None,
    preproc=None,
    title=""
 ):
    """
    Plot recommendations and (optionally) a user profile in the 2D space of a reducer.

    Parameters
    ----------
    X : (n, d) matrix
        Feature matrix for all articles (TF-IDF sparse matrix or dense embeddings).
    reducer : fitted sklearn-like transformer
        Must implement .transform(X). Examples: TruncatedSVD for TF-IDF, PCA for embeddings.
    recs_df : pandas.DataFrame
        Subset of articles_df (recommendations). Its index must match row indices in X.
    v_profile : vector-like, optional
        Profile vector in the same feature space as X (sparse row for TF-IDF, dense for embeddings).
    mode : {'sparse','dense'} or None
        If None, inferred from X being scipy sparse or not.
    preproc : transformer, optional
        Dense preprocessing before reducer (ex: StandardScaler for PCA embeddings).
    """
    is_sparse = _is_sparse_mode(X, mode=mode)
    rng = np.random.default_rng(random_state)

    n = X.shape[0]
    if sample_bg is not None and sample_bg < n:
        bg_idx = rng.choice(n, size=int(sample_bg), replace=False)
        X_bg = X[bg_idx]
    else:
        X_bg = X

    X_bg_p = _prep_X(X_bg, is_sparse=is_sparse, preproc=preproc)
    Z_bg = reducer.transform(X_bg_p)

    rec_idx = recs_df.index.to_numpy()
    X_recs = X[rec_idx]
    X_recs_p = _prep_X(X_recs, is_sparse=is_sparse, preproc=preproc)
    Z_recs = reducer.transform(X_recs_p)

    Z_prof = None
    v_prof_p = _prep_v(v_profile, is_sparse=is_sparse, preproc=preproc)
    if v_prof_p is not None:
        Z_prof = reducer.transform(v_prof_p)

    ix = int(pc_x) - 1
    iy = int(pc_y) - 1
    if ix < 0 or iy < 0:
        raise ValueError("pc_x and pc_y are 1-indexed (PC1=1)")
    if ix >= Z_bg.shape[1] or iy >= Z_bg.shape[1]:
        raise ValueError("Requested PCs exceed reducer output dimensions")

    plt.figure()
    plt.scatter(Z_bg[:, ix], Z_bg[:, iy], s=8, alpha=0.25, label="Articles (fond)", color="blue")
    plt.scatter(Z_recs[:, ix], Z_recs[:, iy], s=30, alpha=0.9, label="Recommandés", color="yellow")
    if Z_prof is not None:
        plt.scatter(Z_prof[0, ix], Z_prof[0, iy], s=120, label="Profil", color="red")
    plt.title(f"Espace reducer (PC{pc_x} vs PC{pc_y}) | {title}")
    plt.xlabel(f"PC{pc_x}")
    plt.ylabel(f"PC{pc_y}")
    plt.legend()
    plt.show()

    # ---- optional annotation
    if annotate_top:
        if annotate_top is True:
            n_annot = 5
        else:
            n_annot = int(annotate_top)
        if n_annot > 0 and ("title" in recs_df.columns):
            plt.figure()
            plt.scatter(Z_bg[:, ix], Z_bg[:, iy], s=8, alpha=0.15)
            plt.scatter(Z_recs[:, ix], Z_recs[:, iy], s=40, alpha=0.9)
            if Z_prof is not None:
                plt.scatter(Z_prof[0, ix], Z_prof[0, iy], s=160, marker="X")
            for j in range(min(n_annot, Z_recs.shape[0])):
                plt.text(Z_recs[j, ix], Z_recs[j, iy], str(recs_df.iloc[j]["title"])[:30] + "…")
            plt.title(f"Espace reducer (PC{pc_x} vs PC{pc_y}) — top annoté")
            plt.xlabel(f"PC{pc_x}")
            plt.ylabel(f"PC{pc_y}")
            plt.show()

def plot_comparaison(
    articles_df,
    X_emb,              # (n_articles, 384) embeddings MiniLM alignés avec articles_df
    scaler, pca,        # scaler + PCA fit sur embeddings
    recs_tfidf_df=None, # df recos TF-IDF (doit contenir id_col)
    recs_emb_df=None,   # df recos Embeddings (doit contenir id_col)
    profile_emb=None,   # optionnel: (384,) ou (1,384)
    est_prof=True, # si True, profile_emb est un vecteur profil utilisateur
    id_col="id",
    pc_x=1, pc_y=2,     # 1-indexed
    title="",
    sample_bg=12000,
    annotate_top=None,
    random_state=42
):
    rng = np.random.default_rng(random_state)

    # --- 0) sécurités
    if id_col not in articles_df.columns:
        raise ValueError(f"articles_df doit contenir la colonne '{id_col}'")

    n = len(articles_df)
    if X_emb.shape[0] != n:
        raise ValueError(f"X_emb doit être aligné avec articles_df: X_emb.shape[0]={X_emb.shape[0]} vs n={n}")

    # mapping id -> index (position dans X_emb)
    ids_all = articles_df[id_col].tolist()
    id_to_pos = {k: i for i, k in enumerate(ids_all)}

    def ids_to_pos(recs_df):
        if recs_df is None:
            return np.array([], dtype=int), []
        if id_col not in recs_df.columns:
            raise ValueError(f"recs_df doit contenir la colonne '{id_col}'")
        ids = recs_df[id_col].tolist()
        pos, missing = [], []
        for _id in ids:
            if _id in id_to_pos:
                pos.append(id_to_pos[_id])
            else:
                missing.append(_id)
        return np.array(pos, dtype=int), missing

    pos_tfidf, miss_tfidf = ids_to_pos(recs_tfidf_df)
    pos_emb,   miss_emb   = ids_to_pos(recs_emb_df)

    if miss_tfidf:
        print(f"[WARN] {len(miss_tfidf)} recos TF-IDF introuvables dans articles_df (ids non matchés).")
    if miss_emb:
        print(f"[WARN] {len(miss_emb)} recos Embeddings introuvables dans articles_df (ids non matchés).")

    # --- 1) Projection PCA de tous les embeddings (ou fond échantillonné)
    ix = pc_x - 1
    iy = pc_y - 1

    if sample_bg is not None and sample_bg < n:
        bg_pos = rng.choice(n, size=sample_bg, replace=False)
    else:
        bg_pos = np.arange(n)

    X_bg = np.asarray(X_emb[bg_pos], dtype=np.float32)
    Z_bg = pca.transform(scaler.transform(X_bg))  # (bg, k)

    # --- 2) Projection PCA des recos (via leurs embeddings)
    Z_tfidf = None
    if pos_tfidf.size > 0:
        X_t = np.asarray(X_emb[pos_tfidf], dtype=np.float32)
        Z_tfidf = pca.transform(scaler.transform(X_t))

    Z_emb = None
    if pos_emb.size > 0:
        X_e = np.asarray(X_emb[pos_emb], dtype=np.float32)
        Z_emb = pca.transform(scaler.transform(X_e))

    # --- 3) Projection profil optionnel
    Z_prof = None
    prof_vec = None  # (384,) en float32 si dispo
    if profile_emb is not None:
        pe = np.asarray(profile_emb, dtype=np.float32)
        if pe.ndim == 2 and pe.shape[0] == 1:
            pe = pe.reshape(-1)
        if pe.ndim != 1:
            raise ValueError("profile_emb doit être de forme (384,) ou (1,384)")
        prof_vec = pe
        Z_prof = pca.transform(scaler.transform(prof_vec.reshape(1, -1)))  # (1, k)

    # --- 3bis) Distances moyennes (dans l'espace embedding ORIGINAL, pas PCA 2D)
    # Référence = profil si dispo, sinon "centralité" = moyenne des embeddings du fond (bg_pos)
    ref_vec = None
    if prof_vec is not None:
        ref_vec = prof_vec.astype(np.float32)
        if est_prof:
            ref_label = "from user profile"
        else:
            ref_label = "from ref article"
    else:
        ref_vec = np.asarray(X_emb[bg_pos], dtype=np.float32).mean(axis=0)
        ref_label = "from centrality"

    def mean_l2_dist(ref, idx_pos):
        if idx_pos is None or len(idx_pos) == 0:
            return None
        X_sel = np.asarray(X_emb[idx_pos], dtype=np.float32)
        d = np.linalg.norm(X_sel - ref.reshape(1, -1), axis=1)
        return float(d.mean())

    dist_tfidf = mean_l2_dist(ref_vec, pos_tfidf)
    dist_emb   = mean_l2_dist(ref_vec, pos_emb)

    dist_tfidf_str = "NA" if dist_tfidf is None else f"{dist_tfidf:.3f}"
    dist_emb_str   = "NA" if dist_emb is None else f"{dist_emb:.3f}"

    title_main = (
        f"Espace PCA embeddings (PC{pc_x} vs PC{pc_y}){title}"
        f"\nDist TF-IDF : {dist_tfidf_str}, Dist Embed : {dist_emb_str} ({ref_label})"
    )

    # --- 4) Plot
    plt.figure()
    plt.scatter(Z_bg[:, ix], Z_bg[:, iy], s=8, alpha=0.20, label="Articles (fond)")

    if Z_tfidf is not None:
        plt.scatter(Z_tfidf[:, ix], Z_tfidf[:, iy], s=45, alpha=0.95, label="Recommandés TF-IDF", color="yellow")

    if Z_emb is not None:
        plt.scatter(Z_emb[:, ix], Z_emb[:, iy], s=60, alpha=0.95, label="Recommandés Embeddings", color="orange")

    if Z_prof is not None:
        if est_prof:
            plt.scatter(Z_prof[0, ix], Z_prof[0, iy], s=100, label="Profil", color="red")
        else:
            plt.scatter(Z_prof[0, ix], Z_prof[0, iy], s=100, label="Article de référence", color="white")

    plt.title(title_main)
    plt.xlabel(f"PC{pc_x}")
    plt.ylabel(f"PC{pc_y}")
    plt.legend()
    plt.show()

    # --- 5) Annotation optionnelle (top N de chaque liste)
    if annotate_top and (recs_tfidf_df is not None or recs_emb_df is not None) and ("title" in articles_df.columns):
        plt.figure()
        plt.scatter(Z_bg[:, ix], Z_bg[:, iy], s=8, alpha=0.12)

        if Z_tfidf is not None:
            plt.scatter(Z_tfidf[:, ix], Z_tfidf[:, iy], s=55, alpha=0.95, marker="o", color="yellow")
            if recs_tfidf_df is not None and id_col in recs_tfidf_df.columns:
                for j, _pos in enumerate(pos_tfidf[:annotate_top]):
                    t = str(articles_df.iloc[_pos].get("title", ""))[:28] + "…"
                    plt.text(Z_tfidf[j, ix], Z_tfidf[j, iy], t)

        if Z_emb is not None:
            plt.scatter(Z_emb[:, ix], Z_emb[:, iy], s=70, alpha=0.95, marker="^", color="orange")
            if recs_emb_df is not None and id_col in recs_emb_df.columns:
                for j, _pos in enumerate(pos_emb[:annotate_top]):
                    t = str(articles_df.iloc[_pos].get("title", ""))[:28] + "…"
                    plt.text(Z_emb[j, ix], Z_emb[j, iy], t)

        if Z_prof is not None:
            plt.scatter(Z_prof[0, ix], Z_prof[0, iy], s=160, marker="X", color="red")

        plt.title(title_main + " — annoté")
        plt.xlabel(f"PC{pc_x}")
        plt.ylabel(f"PC{pc_y}")
        plt.show()
