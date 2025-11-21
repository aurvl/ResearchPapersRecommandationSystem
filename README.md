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
│  ├─ articles_sample_200.csv
│  └─ profile_keywords_template.csv
│
├─ models/
│  └─ tfidf_vectorizer.joblib   # plus tard: vectorizer sauvegarde
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
│  ├─ app.py                 # FastAPI app
│  ├─ templates/
│  │  ├─ img/                # images used in html pages
│  │  ├─ base.html
│  │  ├─ index.html          # page 1 : preferences + bouton Explore
│  │  ├─ explore.html        # page 2 : reco + barre de recherche
│  │  └─ article.html        # page 3 : fiche article + similaires
│  │
│  └─ static/
│     ├─ styles.css
│     ├─ explore.js
│     └─ article.js
│
└─ notebooks/
   └─ 01_exploration.ipynb     # EDA articles + tests TF-IDF / recos
```

## Requirments and tools

## How to run?

## Contributing
