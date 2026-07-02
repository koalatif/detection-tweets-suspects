FROM python:3.11-slim

WORKDIR /app

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code + artefacts.
# NB : models/ est suivi par DVC (donc absent d'un clone Git frais).
#      Avant `docker build` depuis un clone, lancer `dvc pull` pour récupérer
#      models/model.joblib et models/vectorizer.joblib.
COPY . .

EXPOSE 8501

# Healthcheck en Python pur (curl n'est pas présent dans python:slim)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)" || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", \
            "--server.port=8501", "--server.address=0.0.0.0"]
