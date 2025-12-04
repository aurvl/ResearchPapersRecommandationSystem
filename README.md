# RecSys Research Papers

## Subtitle

---

## Description

## What in this repositiory?

```text
├─ README.md
├─ requirements.txt
├─ .gitignore
│
├─ data/
│  ├─ cache/                   # cache des articles en tendance
│  ├─ collect_articles.py      # script de collecte des articles
│  ├─ articles_sample.csv
│  └─ profile_keywords.csv
│
├─ models/
│  └─ tfidf_vectorizer.joblib  # vectorizer sauvegardé
│
├─ src/
│  ├─ __init__.py
│  ├─ config.py                # chemins, constantes (alpha, top_k...)
│  ├─ data_loading.py          # charge les CSV
│  ├─ text_vectorizer.py       # fit / load TF-IDF
│  ├─ profile_builder.py       # construit le profil user
│  ├─ recommender.py           # logique des recos + feedback
│  └─ utils.py                 # fonctions diverses (logging, nettoyage...)
│
├─ app/
│  ├─ api.py                 # FastAPI app
│  ├─ templates/
│  │  ├─ img/                # images used in html pages
│  │  ├─ base.html
│  │  ├─ index.html          # page 1 : preferences + bouton Explore
│  │  ├─ explore.html        # page 2 : reco + barre de recherche
│  │  └─ article.html        # page 3 : fiche article + similaires
│  │
│  └─ static/
│     ├─ img/                # images used in html pages
│     ├─ styles.css
│     ├─ explore.js
│     ├─ article.js
│     └─ search.js
│
└─ notebooks/
   └─ 01_exploration.ipynb     # EDA articles + tests TF-IDF / recom
```

## Requirments and tools

## How to run?

## Contributing

to serigne diop
