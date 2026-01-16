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

Prérequis : Python 3.10+ (recommandé : 3.11).

```bash
# 1) Créer et activer un environnement virtuel
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 2) Installer les dépendances
pip install -r requirements.txt

# 3a) Lancer la démonstration CLI
python main.py

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
│  ├─ profile_keywords.csv    # Mots-clés de profil (mapping option → keywords)
│  ├─ payload.json            # Exemple d’article externe (JSON)
│  └─ cache/                  # Caches arXiv/trends (json + joblib)
│
├─ models/                    # Caches de modèles / représentations
│  ├─ article_embeddings_minilm.joblib
│  ├─ tfidf_vectorizer.joblib
│  └─ X_tfidf.joblib
│
├─ src/                       # Pipeline (chargement, embeddings, reco, trends)
│  ├─ config.py
│  ├─ data_loading.py
│  ├─ embeddings.py
│  ├─ get_trends.py
│  ├─ profile_builder.py
│  ├─ recommender.py
│  ├─ text_vectorizer.py
│  └─ utils.py
│
├─ notebooks/                 # Analyses exploratoires
├─ main.py                    # Script de démonstration (TF‑IDF + embeddings)
└─ requirements.txt
```

## Installation

### Versions

- Python : 3.10+ (testé sur Python 3.11)

### Environnement virtuel

```bash
python -m venv .venv
```

Activation :

- PowerShell :

```powershell
.venv\Scripts\Activate.ps1
```

- cmd.exe :

```bat
.venv\Scripts\activate.bat
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

