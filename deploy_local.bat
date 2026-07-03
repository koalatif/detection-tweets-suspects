@echo off
echo ==========================================
echo Deploiement Local - Detection de Tweets
echo ==========================================

echo [1/3] Recuperation des artefacts DVC...
dvc pull

echo [2/3] Construction de l'image Docker...
docker build -t tweet-detector .

echo [3/3] Lancement du conteneur...
echo L'application sera disponible sur http://localhost:8501
docker run -p 8501:8501 tweet-detector
