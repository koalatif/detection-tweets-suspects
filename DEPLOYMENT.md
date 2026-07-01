# Guide de déploiement cloud

L'application peut être déployée sur trois plateformes sans serveur à gérer.

## Option 1 — Streamlit Community Cloud (le plus simple)
1. Pousser le dépôt sur GitHub.
2. Aller sur https://share.streamlit.io → **New app**.
3. Sélectionner le dépôt, la branche `main`, et le fichier `app/streamlit_app.py`.
4. Déployer. Les dépendances sont lues depuis `requirements.txt`.

> Les artefacts (`models/model.joblib`, `models/vectorizer.joblib`) doivent être présents
> dans le dépôt, ou récupérés au démarrage via `dvc pull` (ajouter un packages.txt/hook).

## Option 2 — Hugging Face Spaces
1. Créer un Space de type **Streamlit**.
2. Copier `app.py` (fourni) à la racine, ou pointer sur `app/streamlit_app.py`.
3. Ajouter le front-matter suivant en tête du `README.md` du Space :

```yaml
---
title: Detection Tweets Suspects
emoji: 🐦
colorFrom: blue
colorTo: red
sdk: streamlit
app_file: app/streamlit_app.py
pinned: false
---
```

## Option 3 — Docker (n'importe quel cloud : Cloud Run, Render, Fly.io, EC2)
```bash
docker build -t tweet-detector .
docker run -p 8501:8501 tweet-detector
# → http://localhost:8501
```

Déploiement sur Google Cloud Run :
```bash
gcloud run deploy tweet-detector --source . --port 8501 --allow-unauthenticated
```
