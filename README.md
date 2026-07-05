---
title: Detection Tweets Suspects
emoji: 🐦
colorFrom: blue
colorTo: red
sdk: streamlit
app_file: app/streamlit_app.py
pinned: false
---
# 🐦 Détection de Tweets Suspects

Solution complète de **classification automatique de tweets suspects**, couvrant tout le
cycle de vie d'un projet de Machine Learning : exploration, prétraitement, versionnement
des données avec **DVC**, représentation vectorielle, comparaison de modèles, optimisation
et **déploiement** via une application Streamlit.

## 📊 Résultats

Modèle retenu : **Régression Logistique (TF-IDF + class weights, C=10)**.

| Métrique   | Score |
|------------|-------|
| Accuracy   | 0.978 |
| Precision  | 0.985 |
| Recall     | 0.990 |
| F1-score   | 0.988 |
| ROC-AUC    | 0.953 |

## 🗂️ Structure du projet

```
.
├── data/
│   ├── raw/tweets_suspect.csv      # dataset versionné par DVC
│   └── processed/                  # sorties du pipeline (générées)
├── src/
│   ├── preprocess.py               # nettoyage du texte (Partie 1)
│   ├── eda.py                      # analyse exploratoire + figures
│   ├── featurize.py                # TF-IDF + split (Parties 3 & 5)
│   ├── train.py                    # entraînement (Partie 4)
│   ├── evaluate.py                 # métriques, confusion, ROC (Partie 6)
│   ├── compare_models.py           # comparaison de 5 modèles
│   └── optimize.py                 # GridSearch + stratégies de déséquilibre
├── notebooks/01_exploration.ipynb  # analyse exploratoire
├── app/streamlit_app.py            # déploiement (Partie 7)
├── models/                         # vectoriseur + modèle (DVC)
├── reports/                        # métriques, figures, rapport PDF
├── dvc.yaml / params.yaml          # pipeline DVC reproductible (Partie 2)
└── requirements.txt
```

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

## 🔁 Reproduire les résultats (DVC)

```bash
dvc pull      # récupère les données/artefacts depuis le remote
dvc repro     # exécute preprocess → featurize → train → evaluate
dvc metrics show
```

Le pipeline est défini dans `dvc.yaml`. Les hyperparamètres sont centralisés dans
`params.yaml` — modifier un paramètre puis relancer `dvc repro` ne réexécute que les
étapes impactées.

## 🧪 Comparaison des modèles (hors pipeline)

```bash
python src/featurize.py
python src/compare_models.py LogisticRegression
python src/compare_models.py MultinomialNB
python src/compare_models.py LinearSVC
python src/compare_models.py RandomForest
python src/compare_models.py XGBoost
python src/optimize.py            # GridSearch + SMOTE/class weights
python src/plot_comparison.py
```

## 🚀 Déploiement — Application Streamlit

```bash
streamlit run app/streamlit_app.py
```

L'interface permet de saisir un tweet, d'obtenir la prédiction (suspect / non suspect)
et la probabilité associée.

## 🧠 Méthodologie

- **Prétraitement** : minuscules, suppression des URLs / mentions / hashtags / ponctuation,
  retrait des *stop words*, stemming de Porter.
- **Représentation** : TF-IDF (uni + bigrammes, 20 000 features, `sublinear_tf`).
- **Déséquilibre** (~90/10) : comparaison *aucune* / *class weights* / *SMOTE*.
  Les **class weights** offrent le meilleur compromis (F1 = 0.988, rappel classe 0 = 0.87).
- **Modèles évalués** : Logistic Regression, Multinomial Naive Bayes, LinearSVC,
  Random Forest, XGBoost (validation croisée 3-fold + jeu de test).
- **Optimisation** : GridSearch sur `C` → optimum `C=10`.
- **Évaluation** : Accuracy, Precision, Recall, F1, matrice de confusion, courbe ROC / AUC.

## ⭐ Extensions bonus

- **MLflow** — suivi d'expériences en complément de DVC (`src/train_mlflow.py`).
  Lancer : `python src/train_mlflow.py` puis `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
- **CI/CD (GitHub Actions)** — `.github/workflows/ci.yml` : lint (flake8), tests `pytest` (8 tests),
  vérification du pipeline DVC à chaque push.
- **Déploiement cloud** — `Dockerfile` + `DEPLOYMENT.md` (Streamlit Cloud / Hugging Face Spaces /
  Docker / Cloud Run). Build : `docker build -t tweet-detector . && docker run -p 8501:8501 tweet-detector`.
- **Transformers avancés (BERT)** — `src/train_bert.py` : embeddings Sentence-BERT
  (`all-MiniLM-L6-v2`) + classifieur, alternative sémantique à TF-IDF.
- **Dashboard de monitoring** — 2ᵉ page Streamlit (`app/pages/`) : suivi des prédictions,
  distributions et **détection de dérive (drift)**.

## ✅ Tests

```bash
pytest tests -v
```

## 🛠️ Stack

Python · pandas · scikit-learn · imbalanced-learn · XGBoost · NLTK · Streamlit · **DVC** · Git · MLflow · Docker · GitHub Actions · pytest

## 📄 Rapport

Le rapport complet est disponible dans `reports/Rapport_Detection_Tweets_Suspects.pdf`.
