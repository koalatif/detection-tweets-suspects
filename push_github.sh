#!/usr/bin/env bash
#
# push_github.sh — Publier le projet sur GitHub avec un remote DVC cloud.
#
# Usage :
#   1. Édite les 3 variables ci-dessous (GITHUB_URL, DVC_REMOTE_TYPE, DVC_REMOTE_URL)
#   2. Rends le script exécutable :  chmod +x push_github.sh
#   3. Lance-le :                    ./push_github.sh
#
# Le script est *idempotent* : tu peux le relancer sans risque.
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# 1. À CONFIGURER
# ─────────────────────────────────────────────────────────────────────────────
# URL de TON dépôt GitHub (crée d'abord un dépôt VIDE sur github.com)
GITHUB_URL="https://github.com/koalatif/detection-tweets-suspects.git"

# Type de remote DVC : "gdrive" (Google Drive), "s3" (AWS), ou "local" (dossier)
DVC_REMOTE_TYPE="gdrive"

# Destination du remote DVC :
#   - gdrive : l'ID du dossier Google Drive (partie après /folders/ dans l'URL)
#   - s3     : s3://mon-bucket/chemin
#   - local  : /chemin/absolu/vers/un/dossier
DVC_REMOTE_URL="gdrive://1TrJA1lM8Qa9BJUkhDVAe2cpa9HHeQG6Q"

BRANCH="main"   # nom de la branche à pousser

# ─────────────────────────────────────────────────────────────────────────────
# 2. NETTOYAGE LOCAL (artefacts lourds, non versionnés)
# ─────────────────────────────────────────────────────────────────────────────
echo "==> Nettoyage des artefacts locaux lourds…"
rm -rf notebooks/.venv .venv mlruns mlartifacts .pytest_cache \
       ./**/__pycache__ ./__pycache__ 2>/dev/null || true
rm -f mlflow.db ./*.tex ./*.log ./*.aux ./*.toc ./*.out 2>/dev/null || true
echo "    OK."

# ─────────────────────────────────────────────────────────────────────────────
# 3. GIT : init si besoin, remote, commit
# ─────────────────────────────────────────────────────────────────────────────
if [ ! -d .git ]; then
  echo "==> Initialisation de Git…"
  git init -q
fi

# Branche principale nommée $BRANCH
git branch -M "$BRANCH" 2>/dev/null || true

# Configure (ou met à jour) le remote 'origin'
if git remote | grep -q '^origin$'; then
  git remote set-url origin "$GITHUB_URL"
else
  git remote add origin "$GITHUB_URL"
fi
echo "==> Remote GitHub : $GITHUB_URL"

echo "==> Commit des changements…"
git add -A
git commit -q -m "Publication du projet (code, pipeline DVC, app, rapport)" || \
  echo "    (rien de nouveau à committer)"

# ─────────────────────────────────────────────────────────────────────────────
# 4. DVC : configurer le remote cloud puis pousser les données/modèles
# ─────────────────────────────────────────────────────────────────────────────
echo "==> Configuration du remote DVC ($DVC_REMOTE_TYPE)…"
case "$DVC_REMOTE_TYPE" in
  gdrive)
    pip install -q "dvc[gdrive]" 2>/dev/null || true ;;
  s3)
    pip install -q "dvc[s3]" 2>/dev/null || true ;;
esac

# (re)définit le remote par défaut nommé 'storage'
dvc remote add -d -f storage "$DVC_REMOTE_URL"
echo "==> Envoi des données/modèles vers le remote DVC…"
dvc push

# ─────────────────────────────────────────────────────────────────────────────
# 5. PUSH GITHUB
# ─────────────────────────────────────────────────────────────────────────────
echo "==> Push vers GitHub (branche $BRANCH)…"
git push -u origin "$BRANCH"

echo ""
echo "✅ Terminé !"
echo "   • Code + pipeline    → $GITHUB_URL"
echo "   • Données + modèles  → $DVC_REMOTE_URL (via dvc)"
echo "   • La CI/CD se lancera automatiquement (onglet « Actions » sur GitHub)."
echo ""
echo "   Un collègue reproduira tout avec :"
echo "       git clone $GITHUB_URL && cd detection-tweets-suspects"
echo "       pip install -r requirements.txt"
echo "       dvc pull && dvc repro"
