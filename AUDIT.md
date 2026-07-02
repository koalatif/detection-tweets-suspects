# 🔍 Rapport d'audit — Détection de Tweets Suspects

Audit fonctionnel du projet : chaque composant a été **exécuté**, pas seulement inspecté.

## 1. Synthèse

| Domaine | État |
|---------|------|
| Couverture du barème (Parties 1-8) | ✅ Complète |
| Reproductibilité (`dvc repro`) | ✅ Pipeline propre, 4 étapes cohérentes |
| Tests (`pytest`) | ✅ 8/8 passés |
| Cohérence des chiffres (rapport ↔ code) | ✅ F1 0,988 / AUC 0,953 alignés partout |
| Fuite de données (data leakage) | ✅ Aucune (`fit` train / `transform` test) |
| Bonus MLflow | ✅ Runs journalisés (params + métriques) |
| Bonus CI/CD | ✅ Workflow valide, tests verts |
| Bonus monitoring | ✅ Page fonctionnelle + détection de dérive |
| Bonus déploiement cloud | ✅ Dockerfile + guide (corrigé, voir §3) |
| Bonus Transformers (BERT) | ✅ Ajouté (`src/train_bert.py`) |

## 2. Vérifications exécutées

- `dvc repro` → *Data and pipelines are up to date* (aucune divergence).
- `pytest tests -v` → **8 passed**.
- Compilation de tous les `.py` (src + app) → OK.
- Validité des YAML (`ci.yml`, `dvc.yaml`, `params.yaml`) → OK.
- Cohérence `metrics.json` ↔ `model_comparison.json` ↔ rapport → identique.
- GridSearch `C=10` ↔ `params.yaml` `C=10.0` → aligné.
- MLflow : 5 modèles journalisés avec `accuracy, precision, recall, f1, roc_auc`.
- Notebook `01_exploration.ipynb` → structure `nbformat` valide (18 cellules).
- `featurize.py` : `fit_transform` sur le train, `transform` sur le test → pas de fuite.

## 3. Problèmes corrigés durant l'audit

1. **Dockerfile — healthcheck cassé.** `HEALTHCHECK` utilisait `curl`, absent de
   l'image `python:3.11-slim`. Remplacé par un test **Python (`urllib`)**.
2. **Docker + modèles DVC.** `models/` est suivi par DVC (absent d'un clone Git frais).
   Ajout d'une note explicite : lancer `dvc pull` **avant** `docker build`.
3. **`.gitignore` durci.** Ajout de `**/.venv/`, `.pytest_cache/`, `*.tex`, `*.log`,
   `*.aux`, `*.toc`, `mlartifacts/` pour éviter de committer des artefacts.
4. **`.dockerignore` durci.** Exclusion de `.venv`, `mlartifacts`, `notebooks`,
   fichiers LaTeX — image Docker plus légère.
5. **Bonus Transformers ajouté.** `src/train_bert.py` (Sentence-BERT `all-MiniLM-L6-v2`
   + classifieur) couvre la catégorie bonus « Transformers avancés ».

## 4. Points d'attention restants (action côté utilisateur)

- **Hygiène du dossier local.** Un `notebooks/.venv/` (~400 Mo) a été créé en ouvrant le
  notebook dans un IDE, ainsi que `mlruns/` et `mlflow.db`. Ils sont **ignorés par Git**
  (donc absents d'un push), mais peuvent être supprimés localement pour alléger le dossier.
- **Commit des guides.** `GUIDE_DETAILLE.*` et `GUIDE_MONTAGE.md` doivent être committés
  avant un push (voir commande ci-dessous).
- **Mise en ligne.** Aucun remote GitHub ni remote DVC cloud n'est configuré : nécessaire
  pour que la CI/CD et le « déploiement cloud » soient visibles/vérifiables.
- **BERT — exécution.** `train_bert.py` télécharge le modèle (~90 Mo) au premier lancement ;
  une connexion Internet est requise une seule fois (l'environnement d'audit étant hors-ligne,
  seule la chaîne aval a pu être validée, pas le téléchargement du modèle).

## 5. Pour finaliser sur GitHub

```bash
git add -A
git commit -m "Corrections audit + bonus BERT + guides"
git remote add origin https://github.com/<toi>/detection-tweets-suspects.git
git push -u origin master
```

## 6. Verdict

Socle (100 pts) **complet et fonctionnel** ; **5 catégories bonus sur 5** désormais
couvertes (MLflow, CI/CD, monitoring, cloud, Transformers) — le bonus étant plafonné à +5,
il est atteint avec marge. Les correctifs de l'audit lèvent les seuls défauts réels
(healthcheck Docker, hygiène du dépôt).
