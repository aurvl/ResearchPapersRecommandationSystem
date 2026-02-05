# RecSys d’articles scientifiques (content-based)

Ce dépôt implémente un système de recommandation **content-based** pour des articles scientifiques.
Le cœur du projet repose sur des **embeddings Sentence-Transformers** (MiniLM, 384 dimensions) pour la similarité sémantique, avec des modes de recommandation : articles similaires, recommandations par profil, articles « hot » à partir de tendances arXiv, et similarité pour un article externe fourni en JSON.

Une application de démonstration FastAPI (templates HTML + JS) permet d’explorer le catalogue et de tester les recommandations.

## Fonctionnalités principales

Le code du dépôt implémente notamment :

- Recommandation **similaire à un article du catalogue** (top‑k voisins par similarité cosinus).
- Recommandation **par profil** (préférences/tags → texte de profil → embedding), avec mise à jour possible par **likes** (agrégation de vecteurs).
- Recommandation **hot/trending** : récupération de textes de tendance depuis arXiv (avec cache hebdomadaire), scoring sémantique et score final combinant tendance/récence/citations.
- Recommandation pour un **article hors catalogue** : upload JSON → texte canonique → embedding → ranking vs catalogue.

## Quickstart

Prérequis :

- Git (pour cloner le dépôt)
- Python 3.10+ (recommandé : 3.11)

```bash
# 1) Créer et activer un environnement virtuel
python -m venv .venv

# 1bis) Activer le venv (selon ton shell)

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Windows (Git Bash)
source .venv/Scripts/activate

# Linux
source .venv/bin/activate

# macOS
source .venv/bin/activate

# 2) Installer les dépendances
pip install -r requirements.txt

# 3a) Lancer la démonstration CLI
Exécuter les notebooks pour une exploration plus approfondie :
- `notebooks/01_exploration.ipynb` : exploration du dataset et du système (TF‑IDF + embeddings)
- `notebooks/02_analyse.ipynb` : analyse des recommandations (TF‑IDF vs embeddings)

# 3b) Lancer l’application Web (FastAPI)
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Puis ouvrir : http://127.0.0.1:8000

## Structure du dépôt

```text
.
├─ app/                       # Démo web FastAPI (routes + templates + statiques)
│  ├─ api.py                  # Backend FastAPI (initialisation + endpoints)
│  ├─ templates/              # Pages HTML (index/explore/article/external)
│  └─ static/                 # JS/CSS + images
│
├─ data/                      # Données et caches de tendance
│  ├─ articles_sample.csv     # Exemple de catalogue (CSV)
│  ├─ collect_articles.py     # Script de collecte (OpenAlex) → parquet local
│  ├─ payload.json            # Exemple d’article externe (JSON)
│  ├─ profile_keywords.csv    # Mots-clés de profil (mapping option → keywords)
│  └─ cache/                  # Caches arXiv/trends (json + joblib)
│
├─ models/                    # Caches de modèles / représentations
│  ├─ tfidf_vectorizer.joblib
│  ├─ X_tfidf.joblib
│  ├─ svd_model.joblib
│  ├─ Z_100.npy
│  ├─ article_embeddings_minilm.joblib
│  ├─ pca_minilm.joblib
│  └─ Z_minilm.npy
│
├─ notebooks/                 # Analyses exploratoires
│  ├─ 01_exploration.ipynb    # Exploration du dataset et du système (TF‑IDF + embeddings)
│  └─ 02_analyse.ipynb        # Analyse des recommandations (TF‑IDF vs embeddings)
├─ src/                       # Pipeline (chargement, embeddings, reco, trends)
├─ main.py                    # Script de démonstration (TF‑IDF + embeddings)
└─ requirements.txt
```

## Installation

### Cloner le dépôt

```bash
git clone https://github.com/aurvl/ResearchPapersRecommandationSystem.git
cd ResearchPapersRecommandationSystem
```

### Versions

- Python : 3.10+ (testé sur Python 3.11)

### Environnement virtuel

```bash
python -m venv .venv
```

Activation :

- PowerShell :

```powershell
.\.venv\Scripts\Activate.ps1
```

- Windows (Git Bash) :

```bash
source .venv/Scripts/activate
```

- cmd.exe :

```bat
.venv\Scripts\activate.bat
```

- Linux / macOS :

```bash
source .venv/bin/activate
```

Installation des dépendances :

```bash
pip install -r requirements.txt
```

## Configuration

Les constantes principales sont dans `src/config.py` :

- Chemins de données : `ARTICLES_PATH`, `PROFILE_KEYWORDS_PATH`
- Caches :
   - embeddings du catalogue : `EMBED_CORPUS` → `models/article_embeddings_minilm.joblib`
   - tendances arXiv : `data/cache/arxiv_trends_YYYY_M_wK.json`
   - hot terms : `HOT_TERMS_CACHE_PATH` → `data/cache/hot_terms.joblib`
   - (prototype TF‑IDF) : `TFIDF_VECTORIZER_PATH`, `X_TFIDF_PATH`
- Modèle embeddings : `LLM_URL = "sentence-transformers/all-MiniLM-L6-v2"`
- Paramètres de ranking : `TOP_K_MAIN`, `TOP_K_SIMILAR`, `PROFILE_ALPHA`

### Variables d’environnement

Le code supporte une variable de debug :

- `DEBUG_L2_NORMS=1` : affiche des statistiques de normes L2 (utile pour diagnostiquer normalisation et shapes).

## Données

Tu as 2 options : utiliser le dataset fourni (recommandé pour reproduire la même expérience), ou collecter le tien.

### Option A — Réutiliser nos données (Google Drive)

Les fichiers de données sont disponibles ici :

- https://drive.google.com/drive/folders/1xj1iG3RwSf0PYftbxBOb-UBFU2svqOms

Deux façons de faire :

1) Télécharger puis copier dans `data/`
   - Télécharge le fichier du catalogue (idéalement `articles.parquet`) et colle-le dans `data/` (ex: `data/articles.parquet`).

2) Utiliser les URLs Drive via la config
   - Dans `src/config.py`, tu peux décommenter les lignes `ARTICLES_PATH = "https://drive.google.com/file/d/..."` (CSV ou parquet) selon ton besoin.
   - Recommandation : pour éviter les limitations Drive (confirm download / gros fichiers), le plus fiable reste de télécharger puis mettre le fichier localement dans `data/`.

### Option B — Collecter ton propre dataset (OpenAlex)

Si tu veux reconstruire un corpus à toi :

```bash
python data/collect_articles.py
```

Le script construit un dataset (2010–2025) à partir d’OpenAlex, déduplique par `id`, puis sauvegarde un parquet local (par défaut `data/articles.parquet`).

Ensuite, assure-toi que `ARTICLES_PATH` dans `src/config.py` pointe bien vers ton fichier local.

## Reproduire l’expérience sans ré-entraîner (TF‑IDF / SVD / PCA / embeddings)

Si tu veux **charger les éléments déjà calculés** (au lieu de refit TF‑IDF / recalculer embeddings / refaire SVD/PCA), télécharge `models.zip` depuis le même Google Drive :

- https://drive.google.com/drive/folders/1xj1iG3RwSf0PYftbxBOb-UBFU2svqOms

Puis :

1) Dézippe `models.zip`
2) Copie le contenu extrait dans le dossier `models/` du projet

Tu dois obtenir des fichiers du style : `tfidf_vectorizer.joblib`, `X_tfidf.joblib`, `svd_model.joblib`, `Z_100.npy`, `article_embeddings_minilm.joblib`, `pca_minilm.joblib`, `Z_minilm.npy`.

### Format attendu du CSV

Le chargement se fait via `src/data_loading.load_articles()`. Le CSV doit contenir au minimum :

- `id` (identifiant)
- `title` (titre)
- `abstract` (résumé)
- `field` (domaine/catégorie)

Les colonnes suivantes sont utilisées si présentes (recommandé) :

- `year` (récence, utilisé dans le score hot)
- `cite_nb` (popularité, utilisé dans le score hot)
- `author`, `journal`, `url` (affichage)

Le texte canonique `text` est construit automatiquement comme concaténation `(title + abstract + field)`.

### Article externe (JSON)

Le parsing d’un article externe est réalisé par `src/data_loading.json_to_text(payload)`. Un JSON valide peut inclure :

```json
{
   "title": "...",
   "abstract": "...",
   "summary": "...",
   "field": "...",
   "authors": ["..."],
   "tags": ["..."],
   "categories": ["..."],
   "url": "..."
}
```

Au minimum, il faut suffisamment de contenu textuel (typiquement `title` et/ou `abstract/summary`, idéalement `field`).

## Cache et performance

### Embeddings du catalogue

Le module `src/embeddings.py` charge ou calcule les embeddings du catalogue et les met en cache dans :

- `models/article_embeddings_minilm.joblib`

Au premier lancement (si le cache est absent ou incompatible), l’encodage peut être long. Les lancements suivants réutilisent le cache.

Le cache inclut la liste des `ids` : l’alignement entre l’ordre des articles et la matrice d’embeddings est vérifié au chargement.

### Caches de tendances

Les tendances arXiv sont mises en cache dans `data/cache/` (fichier JSON hebdomadaire) et la liste de hot terms dans `data/cache/hot_terms.joblib`.

## Utilisation

### Démo CLI

Le script `main.py` exécute une démonstration (TF‑IDF et embeddings) :

```bash
python main.py
```

Il montre notamment : profil → recommandations, update par likes, hot articles, et requête externe à partir de `data/payload.json`.

### Application Web (FastAPI)

L’application est définie dans `app/api.py`.

Lancement :

```bash
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Pages (haut niveau) :

- `/` : saisie de préférences (tags)
- `/explore` : recommandations (profil si tags, sinon hot)
- `/article/{article_id}` : fiche article + similaires
- `/external` : upload JSON externe + recommandations

Endpoints JSON (utilisés par le front) :

- `GET /api/search?q=...` : recherche par titre
- `GET /api/tags` : liste de tags
- `GET /api/recommend/similar/{article_id}?top_k=...`
- `GET /api/recommend/hot?top_k=...`
- `POST /api/interact/like` : mise à jour de recommandations après un like
- `POST /api/recommend/profile` : recommandations à partir d’un profil

## Dépannage

### L’encodage embeddings est très long au premier run

Normal : le cache `models/article_embeddings_minilm.joblib` n’existe pas encore. Laissez l’encodage finir une fois.

### Erreurs de shapes / similarité

Le scoring suppose des vecteurs **L2‑normalisés** et des matrices de dimension cohérente (embeddings : `(N, 384)`).
Activez `DEBUG_L2_NORMS=1` pour diagnostiquer.

### Problèmes de dépendances (Torch / transformers)

Assurez-vous d’être dans un venv propre et d’avoir installé `requirements.txt`. Sur Windows, certaines installations peuvent nécessiter une mise à jour de `pip` :

```bash
python -m pip install --upgrade pip
```

## Licence

Aucune licence n’est spécifiée dans ce dépôt.

## Contribuer

Contributions bienvenues (issues/PR). Pour faciliter la revue :

- décrire le problème et un scénario de reproduction,
- garder des changements ciblés,
- ajouter des notes de test (commandes exécutées, résultats observés).

