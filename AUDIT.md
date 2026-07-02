# 🔍 Rapport d'audit — Détection de Tweets Suspects

Deux passes d'audit ont été menées, chaque composant étant **exécuté** (pas seulement lu).

## Passe 2 — Audit bugs / cohérence / MLOps

Après un passage du code dans un formateur automatique (isort/autopep8) et une bascule
de `use_stemming`, l'audit a révélé et corrigé les points suivants.

### 🔴 Bugs bloquants (corrigés)
1. **`dvc.lock` corrompu** (CRLF + fichier tronqué) → `dvc repro/status/pull` échouaient.
   Régénéré proprement (LF, valide) et `.gitattributes` ajouté pour forcer les fins de ligne LF.
2. **App Streamlit cassée** — le formateur avait remonté `from preprocess import clean_text`
   avant `sys.path.append(...src)` → `ModuleNotFoundError`. Ordre rétabli + directive
   `# isort: skip_file` pour empêcher la récidive.
3. **CI / tests cassés** — même bug d'ordre d'import dans les tests. Corrigé de façon durable
   via `pytest.ini` (`pythonpath = src`) + `tests/conftest.py` : les tests ne dépendent plus
   de l'ordre des imports. `pytest` → **8/8 verts** sans PYTHONPATH.

### 🟠 Incohérences (corrigées)
4. **Train/serve skew** — l'app nettoyait avec `use_stemming=False` alors que le modèle est
   entraîné avec stemming. L'app lit désormais `use_stemming` depuis `params.yaml`
   (source unique de vérité) → prétraitement identique entraînement/service.
5. **Rapport ↔ modèle** — `use_stemming` remis à `true` (config d'origine) ; pipeline
   régénéré ; `metrics.json` et rapport de nouveau alignés (F1 0,988 / AUC 0,953).

### 🟡 Robustesse MLOps (corrigée)
6. **Backend matplotlib** — `MPLBACKEND=Agg` posé via variable d'environnement en tête de
   `eda.py`/`evaluate.py` (robuste au ré-ordonnancement d'imports, headless-safe).
7. **Dépendances épinglées** — `requirements.txt` passe de `>=` à `==` sur le cœur testé
   (reproductibilité) ; `altair` ajouté (utilisé par l'app) ; recommandation Python 3.11.
8. **`.gitattributes`** — normalisation LF pour éviter tout futur `dvc.lock` corrompu par CRLF.

### Points restants (recommandés, non bloquants)
- Monitoring : seuils de dérive codés en dur (73,7 ; 0,898) → à calculer/stocker ; le
  `prediction_log.csv` est éphémère en conteneur (prévoir un stockage persistant).
- Dockerfile : exécution en `root` → idéalement un utilisateur non-root.
- Validation de schéma du CSV d'entrée (ex. pandera/great-expectations) absente.
- Couverture de tests à étendre (featurize, absence de fuite de données).

## Passe 1 — Vérifications fonctionnelles (rappel)
- Barème 1-8 couvert ; `dvc repro` reproductible ; cohérence métriques rapport ↔ code ;
  pas de fuite de données (`fit` train / `transform` test) ; 5 bonus couverts
  (MLflow, CI/CD, monitoring, cloud, Transformers/BERT).

## Verdict
Après correctifs : projet **fonctionnel de bout en bout** — pipeline reproductible,
CI verte, app opérationnelle et cohérente avec le modèle, dépendances figées.
