---
output:
  pdf_document: default
  html_document: default
---
# 🧭 Guide de compréhension et de montage — Détection de Tweets Suspects

Ce guide t'accompagne **pas-à-pas**, de zéro jusqu'au déploiement et au monitoring.
Il est pensé pour que tu comprennes *pourquoi* chaque étape existe, pas seulement *comment*
la lancer.

---

## Table des matières
1. [Vue d'ensemble : que fait le projet ?](#1-vue-densemble)
2. [Comment le projet est organisé](#2-organisation-du-projet)
3. [Le fil rouge : de la donnée brute à la prédiction](#3-le-fil-rouge)
4. [Prérequis et installation](#4-prérequis-et-installation)
5. [Montage pas-à-pas](#5-montage-pas-à-pas)
6. [Comprendre et rejouer le pipeline DVC](#6-le-pipeline-dvc)
7. [Comparer les modèles et optimiser](#7-comparer-les-modèles)
8. [Suivi d'expériences avec MLflow](#8-mlflow)
9. [Déploiement — l'application Streamlit](#9-déploiement-streamlit)
10. [Monitoring et détection de dérive](#10-monitoring)
11. [Déploiement cloud (Docker / Streamlit Cloud / HF)](#11-déploiement-cloud)
12. [CI/CD avec GitHub Actions](#12-cicd)
13. [Dépannage (FAQ)](#13-dépannage)

---

## 1. Vue d'ensemble

**Objectif** : classer automatiquement un tweet en `suspect (1)` ou `non suspect (0)`.

Le projet couvre **tout le cycle de vie d'un projet de Machine Learning** :

```
Donnée brute  →  Nettoyage  →  Vectorisation  →  Modèle  →  Évaluation  →  Déploiement  →  Monitoring
   (CSV)          (texte)        (TF-IDF)      (LogReg)     (F1, ROC)      (Streamlit)     (drift)
```

Deux outils garantissent la **reproductibilité** :
- **Git** versionne le *code*.
- **DVC** versionne les *données* et les *modèles* (trop gros pour Git) et orchestre le *pipeline*.

**Idée clé** : n'importe qui peut cloner le dépôt, taper `dvc repro`, et obtenir
exactement les mêmes résultats que toi.

---

## 2. Organisation du projet

```
detection-tweets-suspects/
├── data/
│   ├── raw/tweets_suspect.csv       ← 60 000 tweets (versionné par DVC)
│   └── processed/                   ← sorties générées par le pipeline
├── src/                             ← tout le code Python
│   ├── preprocess.py    (1) nettoyage du texte
│   ├── eda.py           (1) analyse exploratoire + figures
│   ├── featurize.py     (3) TF-IDF + split train/test
│   ├── train.py         (4) entraînement du modèle final
│   ├── evaluate.py      (6) métriques + matrice de confusion + ROC
│   ├── compare_models.py    comparaison des 5 modèles
│   ├── optimize.py          GridSearch + stratégies de déséquilibre
│   └── train_mlflow.py      suivi d'expériences MLflow (bonus)
├── notebooks/01_exploration.ipynb   ← exploration interactive
├── app/
│   ├── streamlit_app.py             ← interface de prédiction
│   └── pages/2_📊_Monitoring.py     ← dashboard de monitoring (bonus)
├── models/                          ← vectoriseur + modèle (versionnés DVC)
├── reports/                         ← métriques, figures, rapport PDF
├── tests/                           ← tests pytest
├── dvc.yaml / params.yaml           ← définition + paramètres du pipeline
├── Dockerfile / DEPLOYMENT.md       ← déploiement cloud (bonus)
├── .github/workflows/ci.yml         ← CI/CD (bonus)
└── requirements.txt
```

Le chiffre `(n)` renvoie à la partie correspondante du sujet.

---

## 3. Le fil rouge

Comprendre le projet, c'est suivre une donnée du début à la fin :

**a) Un tweet brut** — ex : `"@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer ;D"`

**b) Nettoyage** (`preprocess.py`) — minuscules, suppression des URLs/mentions/ponctuation,
retrait des *stop words*, *stemming*. Résultat : `"awww bummer"`.
> *Pourquoi ?* On enlève le bruit (liens, ponctuation) pour ne garder que les mots porteurs de sens.

**c) Vectorisation** (`featurize.py`) — le texte devient un vecteur de nombres via **TF-IDF**.
Chaque mot (et paire de mots) devient une colonne ; la valeur mesure son importance.
> *Pourquoi ?* Un algorithme ne comprend que des nombres, pas du texte.

**d) Modèle** (`train.py`) — une **Régression Logistique** apprend à associer ces vecteurs
au label. On utilise des *class weights* car les classes sont déséquilibrées (~90 % / 10 %).
> *Pourquoi la LogReg ?* Rapide, interprétable, probabilités fiables, aussi performante que XGBoost ici.

**e) Évaluation** (`evaluate.py`) — on mesure F1, ROC-AUC, matrice de confusion sur des
données **jamais vues** (jeu de test).
> *Pourquoi ?* Vérifier que le modèle généralise au lieu de « mémoriser ».

**f) Déploiement** (`streamlit_app.py`) — on emballe le modèle dans une interface web.

**g) Monitoring** — on surveille les prédictions en production pour détecter une **dérive**.

---

## 4. Prérequis et installation

**Prérequis** : Python 3.9+ et Git installés.

```bash
# 1. Se placer dans le dossier du projet
cd detection-tweets-suspects

# 2. (Recommandé) créer un environnement virtuel isolé
python -m venv .venv
# Windows :
.venv\Scripts\activate
# macOS / Linux :
source .venv/bin/activate

# 3. Installer toutes les dépendances
pip install -r requirements.txt
```

> 💡 L'environnement virtuel évite les conflits entre projets. Tu sauras qu'il est actif
> quand `(.venv)` apparaît devant ton invite de commande.

---

## 5. Montage pas-à-pas

### Étape 1 — Récupérer le code
Si le projet est sur GitHub :
```bash
git clone <URL_DU_DEPOT>
cd detection-tweets-suspects
```
Sinon, tu as déjà le dossier localement.

### Étape 2 — Récupérer les données et modèles versionnés
```bash
dvc pull
```
Cette commande télécharge le dataset et les modèles depuis le *remote* DVC.
> ℹ️ Ici le remote est un dossier local. En équipe, ce serait Google Drive / S3.
> Si `dvc pull` n'a rien à faire (données déjà présentes), c'est normal.

### Étape 3 — Vérifier que tout est là
```bash
ls data/raw/          # doit contenir tweets_suspect.csv
dvc status            # état du pipeline
```

### Étape 4 — Tout reconstruire d'une commande
```bash
dvc repro
```
🎉 En une commande, DVC nettoie les données, vectorise, entraîne et évalue le modèle.

---

## 6. Le pipeline DVC

Le pipeline est décrit dans **`dvc.yaml`**. Il enchaîne 4 étapes, chacune avec ses
*dépendances* (entrées) et ses *sorties* :

```
preprocess  →  featurize  →  train  →  evaluate
```

| Étape        | Script            | Entrée               | Sortie                        |
|--------------|-------------------|----------------------|-------------------------------|
| `preprocess` | preprocess.py     | CSV brut             | `data/processed/clean.csv`    |
| `featurize`  | featurize.py      | clean.csv            | matrices TF-IDF + vectoriseur |
| `train`      | train.py          | matrices train       | `models/model.joblib`         |
| `evaluate`   | evaluate.py       | matrices test + modèle | `reports/metrics.json` + figures |

Les **paramètres** (nb de features, C, stratégie de déséquilibre…) sont centralisés dans
**`params.yaml`**. C'est la force de DVC :

```bash
# Change un paramètre dans params.yaml (ex: featurize.max_features: 30000)
# puis :
dvc repro
```
DVC **ne réexécute que les étapes impactées** par le changement — pas besoin de tout relancer.

Commandes utiles :
```bash
dvc dag              # visualise le graphe du pipeline
dvc metrics show     # affiche les métriques du modèle
dvc params diff      # montre ce qui a changé dans les paramètres
```

---

## 7. Comparer les modèles

Le pipeline entraîne le modèle *retenu*. Pour **reproduire la comparaison des 5 modèles**
(Parties 4-6), lance-les à la main après avoir vectorisé :

```bash
python src/featurize.py            # crée les matrices TF-IDF (si pas déjà fait)

python src/compare_models.py LogisticRegression
python src/compare_models.py MultinomialNB
python src/compare_models.py LinearSVC
python src/compare_models.py RandomForest
python src/compare_models.py XGBoost

python src/optimize.py             # GridSearch + comparaison class_weight / SMOTE
python src/plot_comparison.py      # génère les graphiques comparatifs
```

Les résultats s'accumulent dans `reports/model_comparison.json` et les figures dans
`reports/figures/`. **Résultat attendu** : la Régression Logistique et LinearSVC arrivent
en tête (F1 ≈ 0,988).

---

## 8. MLflow

MLflow enregistre chaque expérience (paramètres, métriques, modèle) pour les comparer
visuellement — en **complément** de DVC.

```bash
# 1. Lancer les entraînements suivis
python src/train_mlflow.py

# 2. Ouvrir l'interface web MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
# → ouvre http://localhost:5000 dans ton navigateur
```

Dans l'interface, tu peux trier les 5 runs par F1 ou AUC, comparer les paramètres,
et télécharger les modèles.

---

## 9. Déploiement — Streamlit

L'application permet de saisir un tweet et d'obtenir la prédiction + la probabilité.

```bash
streamlit run app/streamlit_app.py
```
→ ouvre automatiquement `http://localhost:8501`.

**Comment ça marche** : l'app charge `models/vectorizer.joblib` et `models/model.joblib`,
applique **le même nettoyage** qu'à l'entraînement (crucial !), puis affiche la probabilité.
Chaque prédiction est journalisée dans `reports/prediction_log.csv` pour le monitoring.

> ⚠️ Le modèle doit exister (`dvc repro` ou `dvc pull`) **avant** de lancer l'app.

---

## 10. Monitoring

Une **2ᵉ page** apparaît dans la barre latérale de l'app : `📊 Monitoring`.

Elle suit en temps réel :
- le **nombre de prédictions** servies,
- la **distribution** des classes et des probabilités prédites,
- le **volume dans le temps**,
- une **alerte de dérive (drift)** : si la longueur moyenne des tweets entrants ou le taux
  de « suspect » s'éloigne trop des statistiques d'entraînement, un réentraînement est conseillé.

> *Pourquoi surveiller la dérive ?* En production, les données évoluent (nouveaux sujets,
> nouveau vocabulaire). Un modèle figé se dégrade silencieusement : le monitoring le détecte.

Pour la voir se remplir : fais quelques prédictions dans la page principale, puis ouvre
la page Monitoring.

---

## 11. Déploiement cloud

Trois options, détaillées dans **`DEPLOYMENT.md`**. La plus simple d'abord.

### Option A — Docker (universel)
```bash
docker build -t tweet-detector .
docker run -p 8501:8501 tweet-detector
# → http://localhost:8501
```

### Option B — Streamlit Community Cloud (gratuit, zéro serveur)
1. Pousser le dépôt sur GitHub.
2. Aller sur https://share.streamlit.io → **New app**.
3. Choisir le dépôt, la branche `main`, le fichier `app/streamlit_app.py`. Déployer.

### Option C — Google Cloud Run (à partir du Docker)
```bash
gcloud run deploy tweet-detector --source . --port 8501 --allow-unauthenticated
```

---

## 12. CI/CD

À chaque `git push`, GitHub Actions exécute automatiquement
(`.github/workflows/ci.yml`) :
1. installation des dépendances,
2. **lint** (flake8) pour repérer les erreurs de syntaxe,
3. **tests** `pytest` (8 tests unitaires + intégration),
4. vérification du pipeline DVC.

Pour lancer les tests **en local** avant de pousser :
```bash
pytest tests -v
```
> *Pourquoi ?* La CI/CD empêche de merger du code cassé : elle est ta filet de sécurité.

---

## 13. Dépannage

| Problème | Solution |
|----------|----------|
| `ModuleNotFoundError` | L'environnement virtuel est-il activé ? `pip install -r requirements.txt` |
| `dvc pull` ne fait rien | Normal si les données sont déjà présentes localement |
| Streamlit : « model not found » | Lance d'abord `dvc repro` (ou `dvc pull`) pour générer `models/` |
| `git status` montre des fichiers modifiés sans raison | Sur un montage réseau : différences de *permissions* uniquement (0 insertion/suppression). Disparaît sur un disque local. |
| Les nuages de mots / NLTK échouent hors-ligne | Le projet embarque sa propre liste de stop words ; aucun téléchargement requis |
| Port 8501 déjà utilisé | `streamlit run app/streamlit_app.py --server.port 8502` |

---

## Récapitulatif express (checklist)

```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt                     # 1. dépendances
dvc pull                                            # 2. données + modèles
dvc repro                                           # 3. reconstruire le pipeline
dvc metrics show                                    # 4. voir les métriques
pytest tests -v                                     # 5. vérifier les tests
streamlit run app/streamlit_app.py                  # 6. lancer l'app + monitoring
```

Bon montage ! 🚀
