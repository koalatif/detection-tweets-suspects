# Rapport d'optimisation : Modèle BERT vs TF-IDF

Ce document détaille les résultats obtenus suite à l'optimisation des hyperparamètres du classifieur entraîné sur les embeddings générés par le modèle **Sentence-Transformers** (`all-MiniLM-L6-v2`).

## 1. Contexte
Dans l'approche classique (présentée par défaut dans le projet), la représentation textuelle est basée sur une vectorisation **TF-IDF**, couplée à une régression logistique. 

L'approche alternative consiste à utiliser un réseau de neurones pré-entraîné de type Transformer (ici une version allégée de BERT) pour capturer le contexte sémantique des phrases et produire des vecteurs denses (embeddings) de dimension 384. Ces vecteurs alimentent ensuite une régression logistique.

## 2. Optimisation des hyperparamètres
Une recherche sur grille (`GridSearchCV`) a été appliquée à l'hyperparamètre `C` de la régression logistique finale afin de trouver la valeur optimale de régularisation pour les embeddings denses. 
La métrique cible lors de la recherche était le F1-score.

**Meilleurs paramètres trouvés :** `{'C': 10}`

## 3. Comparaison des performances

| Modèle | Représentation | Précision | Rappel | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Régression Logistique (Baseline)** | TF-IDF | 0.985 | 0.990 | **0.988** | **0.953** |
| **Régression Logistique (Optimisé)** | Sentence-BERT | 0.970 | 0.793 | 0.873 | 0.873 |

## 4. Analyse et Recommandations
Les résultats révèlent que pour cette tâche spécifique de détection de tweets suspects, l'approche **TF-IDF classique surpasse significativement l'approche basée sur les embeddings de Sentence-BERT** non affinés (fine-tuning).

### Pourquoi BERT est-il moins performant ici ?
1. **Absence de fine-tuning** : Le modèle Sentence-Transformers est utilisé "out-of-the-box" comme extracteur de caractéristiques (feature extractor). Les poids du réseau de neurones n'ont pas été ré-entraînés spécifiquement sur le jargon ou le type d'anomalies présents dans vos tweets.
2. **Sensibilité aux mots-clés** : La détection de tweets suspects (spam, phishing) repose souvent sur la présence de mots-clés très spécifiques, d'URL ou de motifs explicites. TF-IDF excelle pour ce type de corrélation forte mot/label, tandis que BERT lisse ces signaux dans un vecteur sémantique global.

### Avantages de BERT :
- Capture sémantique du contexte et des synonymes.
- Généralisation souvent meilleure sur des formulations inédites.

### Avantages de TF-IDF :
- Extrêmement rapide à l'inférence.
- Empreinte mémoire très faible (contrairement à BERT qui nécessite le chargement des poids du Transformer).

---
*Généré lors de l'audit automatique du projet.*

## Annexe : Aide-mémoire Git & DVC

Dans ce projet, **Git** versionne le code, tandis que **DVC** s'occupe des données lourdes (fichiers csv, modèles `.joblib`). Voici les commandes principales pour piloter ce projet :

### Reproduire le pipeline
Si vous modifiez un paramètre dans `params.yaml` ou le code d'une étape, DVC sait ce qu'il faut réexécuter :
```bash
dvc repro         # Relance uniquement les étapes nécessaires
dvc repro -f train # Force l'étape d'entraînement
```

### Évaluer les performances
DVC garde un historique des métriques :
```bash
dvc metrics show  # Affiche le tableau des performances (accuracy, f1, etc.)
dvc metrics diff  # Compare la version actuelle avec les anciens commits
```

### Sauvegarder son travail
Lorsqu'un modèle ou des données sont modifiés :
```bash
dvc commit        # 1. Fige les nouvelles données/modèles dans DVC
git add .         # 2. Ajoute les fichiers générés par DVC (dvc.lock) à Git
git commit -m "..." # 3. Sauvegarde le code et le pointeur DVC dans Git
```

### Synchroniser (Cloud & Équipe)
Pour cloner le projet ou le pousser vers un espace de stockage en ligne (GitHub + S3/G-Drive) :
```bash
dvc push          # Envoie les gros fichiers vers le stockage distant (remote)
dvc pull          # Télécharge les gros fichiers depuis le stockage distant
```
