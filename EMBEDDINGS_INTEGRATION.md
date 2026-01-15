# Documentation Technique : Intégration du Modèle Sentence-Transformers

## 1. Objectif de l'ajout

L'intégration du modèle sentence-transformers/all-MiniLM-L6-v2 vise à offrir une représentation vectorielle alternative des documents scientifiques basée sur des embeddings contextuels pré-entraînés. Cette approche complète le système existant en permettant une capture sémantique des textes indépendante de la fréquence des termes.

## 2. Nouveau module : src/embeddings.py

Un module dédié a été créé pour encapsuler les opérations liées aux embeddings. Il contient trois fonctions principales :

**get_embedding_model** - Charge et retourne une instance du modèle SentenceTransformer à partir d'un identifiant Hugging Face.

**encode_texts** - Encode une liste de textes en vecteurs denses de dimension 384. La fonction applique une normalisation L2 optionnelle pour faciliter le calcul de similarité cosinus via produit scalaire.

**get_or_compute_article_embeddings** - Gère le cycle de vie complet des embeddings d'articles : chargement depuis le cache si disponible et compatible, calcul via le modèle si nécessaire, sauvegarde automatique. La compatibilité du cache est vérifiée via le nombre d'articles et la correspondance des identifiants.

## 3. Nouvelles constantes de configuration

Trois constantes ont été ajoutées au fichier src/config.py :

**HF_EMBED_MODEL_NAME** - Identifiant du modèle Hugging Face utilisé (sentence-transformers/all-MiniLM-L6-v2).

**EMBEDDINGS_CACHE_PATH** - Chemin de sauvegarde des embeddings des articles (format joblib).

**PROFILE_EMBED_CACHE_PATH** - Chemin de sauvegarde des embeddings de profils utilisateur (réservé pour usage futur).

## 4. Nouvelles fonctions de recommandation

Trois fonctions ont été ajoutées au module src/recommender.py pour implémenter la logique de recommandation basée sur les embeddings :

**recommend_for_profile_emb** - Calcule la similarité cosinus entre un vecteur de profil utilisateur et les vecteurs d'articles via produit scalaire. Retourne les articles les plus similaires en excluant optionnellement certains identifiants.

**update_profile_with_likes_emb** - Met à jour le profil utilisateur en calculant une combinaison pondérée entre le profil initial et le centroïde des articles likés. Le vecteur résultant est re-normalisé pour maintenir la cohérence des calculs de similarité.

**recommend_similar_to_article_emb** - Identifie les articles similaires à un article de référence en calculant les similarités cosinus entre son vecteur et l'ensemble des vecteurs d'articles.

## 5. Routage dans main.py

La variable REPRESENTATION_MODE contrôle le mode de représentation utilisé. Trois valeurs sont possibles : "tfidf", "embeddings", ou "both".

Lorsque le mode "embeddings" est actif, le flux d'exécution charge le modèle SentenceTransformer, calcule ou récupère les embeddings des articles, encode le texte de profil utilisateur, puis exécute les trois scénarios de recommandation : profil initial, profil mis à jour après likes, et similarité à un article donné.

Les résultats sont affichés dans une section dédiée clairement identifiée pour éviter toute confusion avec les résultats TF-IDF lorsque le mode "both" est utilisé.

## 6. Gestion du cache et du fallback

Le système de cache utilise le format joblib pour stocker les embeddings accompagnés des identifiants d'articles. Avant rechargement, une vérification de compatibilité est effectuée : le cache n'est utilisé que si le nombre d'articles et leurs identifiants correspondent exactement au DataFrame courant.

Un mécanisme de fallback automatique est implémenté : en cas d'échec du chargement du modèle ou du calcul des embeddings (absence de torch, erreur réseau, etc.), le système bascule automatiquement vers le mode TF-IDF avec émission d'un avertissement via loguru. Cette stratégie garantit la robustesse du système.

## 7. Résumé synthétique

L'intégration ajoute un second mode de représentation documentaire basé sur des embeddings neuronaux, accessible via une variable de configuration unique. Le système conserve une architecture modulaire avec séparation stricte des pipelines TF-IDF et embeddings. La gestion intelligente du cache et le mécanisme de fallback assurent la fiabilité opérationnelle du système étendu.
