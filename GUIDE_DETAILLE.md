---
title: "Guide détaillé — Détection de Tweets Suspects"
subtitle: "De la donnée brute au déploiement et au monitoring, expliqué pas à pas"
date: "Projet de Machine Learning"
lang: fr
toc: true
toc-depth: 2
numbersections: true
geometry: margin=2.2cm
---

\newpage

# Introduction : comment lire ce guide

Ce document est une version **approfondie** du guide de montage. Là où le guide court
disait *quoi taper*, celui-ci explique **ce qui se passe réellement** et **pourquoi**, avec
des exemples chiffrés, le décorticage des scripts et les sorties que tu dois observer.

Trois profils de lecture :

- **Tu veux juste faire tourner le projet**  lis les sections 4 à 6, puis la *checklist* finale.
- **Tu veux comprendre pour l'oral / le rapport**  lis tout, en insistant sur les sections 3, 7 et 8.
- **Tu veux réutiliser le projet ailleurs**  sections 6 (DVC), 11 (Docker) et 12 (CI/CD).

> **Convention.** : les blocs `en gris` sont des commandes à taper dans un terminal.
> Les encadrés « » En clair » reformulent une idée technique en langage simple.

\newpage

# Vue d'ensemble : quel problème résout-on ?

## Le problème métier

Sur un réseau social, un flux continu de messages arrive. On veut **filtrer
automatiquement** ceux qui sont *suspects* (spam, arnaques, contenu trompeur) de ceux qui
sont *normaux*, sans lire chaque message à la main. C'est une tâche de **classification
binaire de texte** : à chaque tweet, associer une étiquette `0` (non suspect) ou `1` (suspect).

## Les données

Un fichier `tweets_suspect.csv` de **60 000 lignes**, deux colonnes :

| Colonne   | Type    | Description                                  |
|-----------|---------|----------------------------------------------|
| `message` | texte   | le tweet brut                                |
| `label`   | 0 ou 1  | 1 = suspect, 0 = non suspect                 |

Deux faits importants découverts à l'exploration :

1. **Déséquilibre fort** : ~89,8 % de la classe 1 contre ~10,2 % de la classe 0.
2. **293 doublons** exacts, supprimés au nettoyage (il reste 59 450 tweets).

**» En clair.** : comme 9 tweets sur 10 sont « suspects », un modèle paresseux qui répond
toujours « suspect » aurait déjà 90 % de bonnes réponses. C'est pourquoi l'*accuracy* seule
est trompeuse ici, et pourquoi on soignera le rééquilibrage et des métriques comme le F1.

## La chaîne de traitement

```
Tweet brut  Nettoyage  Vectorisation  Modèle  Évaluation  Déploiement  Monitoring
```

Chaque maillon a un script dédié dans `src/`. Deux outils encadrent le tout :

- **Git** : versionne le *code* (qui a changé quoi, quand).
- **DVC** : versionne les *données* et *modèles* (trop volumineux pour Git) et **orchestre
  le pipeline** pour qu'il soit reproductible d'une seule commande.

\newpage

# Le fil rouge : suivons un tweet de bout en bout

Prenons un tweet réel du dataset et suivons sa transformation.

## Étape a — Le tweet brut
```
@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer. 
You shoulda got David Carr of Third Day to do it. ;D
```

## Étape b — Le nettoyage (`preprocess.py`)
On applique, dans l'ordre :

| Opération                       | Avant  Après                              | Pourquoi                                   |
|---------------------------------|--------------------------------------------|--------------------------------------------|
| Minuscules                      | `Awww`  `awww`                            | `Awww` et `awww` doivent être le même mot  |
| Suppression des URLs            | `http://twitpic.com/...`  *(vide)*        | un lien est unique, aucun signal réutilisable |
| Suppression des mentions `@`    | `@switchfoot`  *(vide)*                   | un pseudo n'aide pas à généraliser         |
| Ponctuation / chiffres / emojis | `;D`, `,`, `.`  *(vide)*                  | on ne garde que des lettres                |
| Stop words                      | `that's`, `of`, `to`  *(vide)*            | mots trop fréquents, sans information       |
| Stemming (Porter)               | `shoulda` reste, `bummer` reste            | on ramène les mots à leur racine           |

**Résultat** : `awww bummer shoulda got david carr third day`

**» En clair.** : on jette tout ce qui est « décoratif » ou propre à ce tweet précis, pour
ne garder que les mots qui pourraient réapparaître ailleurs et porter du sens.

## Étape c — La vectorisation (`featurize.py`)
Le texte nettoyé devient une **liste de nombres**. On utilise **TF-IDF** (détaillé en
section 7). Intuition : chaque mot du vocabulaire devient une colonne ; la valeur dans la
colonne mesure « à quel point ce mot est important dans ce tweet ». Un tweet devient donc
un vecteur de 20 000 nombres (dont la plupart valent 0).

## Étape d — La prédiction (`train.py` / le modèle)
La **Régression Logistique** calcule un score entre 0 et 1 : la *probabilité que le tweet
soit suspect*. Au-dessus de 0,5  classe 1, sinon classe 0.

## Étape e — L'évaluation (`evaluate.py`)
On compare les prédictions du modèle aux vraies étiquettes, sur des tweets **qu'il n'a
jamais vus**, et on calcule des métriques (section 7).

## Étape f/g — Déploiement et monitoring
Le modèle est emballé dans une app web (Streamlit), et on surveille son comportement dans
le temps (dérive).

\newpage

# Prérequis et installation détaillés

## Ce qu'il te faut
- **Python 3.9 ou plus** — vérifie avec `python --version`.
- **Git** — vérifie avec `git --version`.
- ~500 Mo d'espace disque (dépendances + données).

## Pourquoi un environnement virtuel ?
Un *environnement virtuel* est un dossier Python isolé, propre à ce projet. Il évite que les
bibliothèques de ce projet entrent en conflit avec celles d'un autre.

```bash
cd detection-tweets-suspects

python -m venv .venv            # crée l'environnement (dossier .venv/)

# Activation :
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows (PowerShell / CMD)
```
Quand il est actif, ton invite affiche `(.venv)` au début. Pour sortir : `deactivate`.

## Installer les dépendances
```bash
pip install -r requirements.txt
```
Cela installe pandas, scikit-learn, imbalanced-learn, xgboost, nltk, streamlit, dvc, mlflow, etc.

**» En clair.** : `requirements.txt` est la « liste de courses » du projet. Toute personne
qui l'installe obtient exactement les mêmes outils, donc les mêmes résultats.

\newpage

# Montage pas-à-pas (avec sorties attendues)

## Étape 1 — Récupérer le code
```bash
git clone <URL_DU_DEPOT>
cd detection-tweets-suspects
```

## Étape 2 — Récupérer données et modèles (DVC)
```bash
dvc pull
```
*Sortie typique* : `A       data/raw/tweets_suspect.csv` puis `1 file added`. Si tout est déjà
là : `Everything is up to date.`

## Étape 3 — Vérifier l'état
```bash
dvc status
```
`Data and pipelines are up to date.` signifie que rien n'a besoin d'être recalculé.

## Étape 4 — Tout reconstruire
```bash
dvc repro
```
*Sortie typique* (abrégée) :
```
Running stage 'preprocess':  Pretraitement termine : 59450 lignes
Running stage 'featurize':   TF-IDF: 20000 features | train 47560 | test 11890
Running stage 'train':       Modele entraine (logistic, class_weight)
Running stage 'evaluate':    {accuracy: 0.978, f1: 0.988, roc_auc: 0.953}
```
En une commande, les 4 étapes s'enchaînent dans le bon ordre.

## Étape 5 — Voir les métriques
```bash
dvc metrics show
```
```
Path                  accuracy  f1       precision  recall   roc_auc
reports/metrics.json  0.97813   0.98784  0.98544    0.99025  0.95315
```

\newpage

# Le pipeline DVC, expliqué en profondeur

## Le problème que DVC résout
Git est fait pour du texte (code), pas pour des fichiers de 5 Mo ou des modèles binaires.
DVC complète Git : il garde le *code* dans Git et les *gros fichiers* à part, tout en
gardant la trace de quelle version de données va avec quelle version de code.

## Les fichiers clés

- **`data/raw/tweets_suspect.csv.dvc`** — un petit fichier texte (un « pointeur ») qui
  contient l'empreinte (`md5`) du vrai CSV. C'est *lui* qui est dans Git ; le vrai CSV, lui,
  est stocké dans le cache DVC et le *remote*.
  ```yaml
  outs:
    - md5: 60e5f26c964cb0a766047fd993102170
      size: 4695170
      path: tweets_suspect.csv
  ```
- **`dvc.yaml`** — la *recette* du pipeline : les étapes, leurs entrées (`deps`), sorties
  (`outs`), paramètres (`params`) et métriques.
- **`params.yaml`** — tous les réglages au même endroit (nb de features, `C`, stratégie…).
- **`dvc.lock`** — généré automatiquement : il fige les empreintes exactes de chaque étape,
  garantissant une reproduction à l'identique.

## Comment DVC décide quoi recalculer
DVC calcule une empreinte de chaque *dépendance* (script + données + params). Si rien n'a
changé pour une étape, il la **saute**. Si tu modifies par exemple `featurize.max_features`
dans `params.yaml`, DVC voit que `featurize` et *tout ce qui en dépend* (train, evaluate)
doivent être refaits — mais pas `preprocess`.

```bash
# Exemple : augmenter le vocabulaire
# édite params.yaml -> featurize.max_features: 30000
dvc repro          # ne réexécute que featurize -> train -> evaluate
```

## Le graphe du pipeline
```bash
dvc dag
```
```
tweets_suspect.csv.dvc  preprocess  featurize  train  evaluate
```

## Le stockage distant (remote)
```bash
dvc remote list          # montre le remote configuré
dvc push                 # envoie données/modèles vers le remote
dvc pull                 # les récupère
```
**» En clair.** : `git push/pull` déplace le *code*, `dvc push/pull` déplace les *données et
modèles*. Les deux ensemble = projet 100 % reproductible par un collègue.

\newpage

# Les concepts de Machine Learning, expliqués

Cette section est le cœur de la compréhension. Prends ton temps.

## 7.1 TF-IDF avec un exemple chiffré

**TF-IDF** = *Term Frequency × Inverse Document Frequency*. Il transforme un texte en
nombres en donnant à chaque mot un poids = « fréquent dans ce tweet, mais rare ailleurs ».

- **TF** (fréquence du terme) : combien de fois le mot apparaît dans *ce* tweet.
- **IDF** (fréquence documentaire inverse) : `log(N / nb de tweets contenant le mot)`.
  Un mot présent partout (ex. « today ») a un IDF proche de 0  poids faible. Un mot rare a
  un IDF élevé  poids fort.

**Mini-exemple** sur 3 tweets :
```
T1: "free money"        T2: "free game"        T3: "hello world"
```
Vocabulaire : `free, money, game, hello, world`. Le mot `free` apparaît dans 2 tweets sur 3
 IDF modéré. `money` n'apparaît que dans T1  IDF élevé  dans le vecteur de T1, `money`
pèsera **plus** que `free`. C'est ainsi que TF-IDF fait ressortir les mots discriminants.

Dans le projet on ajoute :

- **n-grammes (1,2)** : on garde les mots seuls *et* les paires. `« free money »` devient
  une colonne à part entière — utile car l'expression est plus parlante que chaque mot isolé.
- **`max_features=20000`** : on garde les 20 000 termes les plus utiles (sinon des millions).
- **`min_df=2`** : on ignore les termes qui n'apparaissent qu'une seule fois (bruit).
- **`sublinear_tf`** : on atténue l'effet des répétitions (log) pour éviter qu'un mot répété
  10 fois domine tout.

**» En clair.** : TF-IDF fabrique, pour chaque tweet, une « empreinte numérique » où les
mots caractéristiques ressortent et les mots passe-partout s'effacent.

## 7.2 Le déséquilibre des classes

Rappel : 90 % de classe 1. Trois parades ont été comparées :

- **Rien** : le modèle est biaisé vers la classe majoritaire ; il « rate » les rares tweets 0.
- **Class weights** : on dit au modèle « une erreur sur la classe rare coûte ~9× plus cher ».
  Il fait donc davantage attention à la minorité. *(Retenu.)*
- **SMOTE** : on crée des exemples synthétiques de la classe rare pour rééquilibrer. Ici, cela
  dégrade le F1 car l'espace TF-IDF est très creux (les points synthétiques sonnent faux).

Résultats mesurés :

| Stratégie      | F1 (classe 1) | Rappel classe 0 (minoritaire) |
|----------------|---------------|-------------------------------|
| Aucune         | 0,986         | 0,779                         |
| **Class weights** | **0,988**  | **0,873**                     |
| SMOTE          | 0,974         | 0,846                         |

**» En clair.** : « rappel classe 0 = 0,873 » veut dire qu'on retrouve 87 % des vrais tweets
non-suspects, contre seulement 78 % sans correction. Les class weights améliorent nettement
la détection de la classe rare.

## 7.3 Le split train / test et la validation croisée

- **Train (80 %)** : le modèle apprend dessus.
- **Test (20 %)** : on l'évalue dessus — des tweets **jamais vus**. C'est le seul juge honnête.
- **Split stratifié** : on garde la proportion 90/10 dans les deux ensembles.
- **Validation croisée (3-fold)** : on découpe le train en 3, on entraîne sur 2 et on valide
  sur le 3ᵉ, en tournant. Cela donne une performance *moyenne* plus fiable qu'un seul essai.

**» En clair.** : évaluer un modèle sur des données qu'il a déjà vues, c'est comme donner à
un élève l'examen qu'il a révisé mot pour mot. Le jeu de test, ce sont des questions inédites.

## 7.4 Lire les métriques (avec les chiffres du projet)

La **matrice de confusion** croise vérité et prédiction :

|                     | Prédit 0     | Prédit 1     |
|---------------------|--------------|--------------|
| **Vrai 0**          | Vrais Négatifs | Faux Positifs |
| **Vrai 1**          | Faux Négatifs  | Vrais Positifs |

À partir d'elle :

- **Precision** = parmi les « suspect » prédits, combien le sont vraiment  **0,985**.
- **Recall** (rappel) = parmi les vrais suspects, combien on en attrape  **0,990**.
- **F1** = moyenne harmonique de precision et recall (équilibre les deux)  **0,988**.
- **Accuracy** = proportion globale de bonnes réponses  **0,978** *(trompeuse ici à cause du
  déséquilibre)*.
- **ROC-AUC** = capacité du modèle à *classer* un suspect au-dessus d'un non-suspect, quel que
  soit le seuil  **0,953**. Une valeur de 0,5 = hasard, 1,0 = parfait.

**» En clair.** : l'AUC de 0,953 est la métrique la plus rassurante : elle prouve qu'au-delà de
l'effet « 90 % de la même classe », le modèle capte un **vrai signal** dans le texte.

## 7.5 Le GridSearch (optimisation)

On teste automatiquement plusieurs réglages et on garde le meilleur (au sens du F1 en
validation croisée). Ici on a balayé le paramètre de régularisation `C ∈ {0.1, 1, 10, 100}`.

- `C` **petit** = régularisation forte = modèle simple (risque de sous-apprentissage).
- `C` **grand** = régularisation faible = modèle plus libre (risque de sur-apprentissage).

**Optimum trouvé : `C = 10`.**

\newpage

# Décorticage des scripts

## `preprocess.py` — le nettoyage
Points clés :
```python
STOPWORDS = set("a about above ... your yours".split())  # liste embarquée (hors-ligne)

def clean_text(text, use_stemming=True):
    text = str(text).lower()                 # minuscules
    text = _url_re.sub(" ", text)            # enlève les URLs
    text = _mention_re.sub(" ", text)        # enlève les @mentions
    text = text.replace("#", " ")            # enlève les # de hashtags
    text = _nonalpha_re.sub(" ", text)       # ne garde que les lettres
    tokens = [t for t in text.split()
              if t not in STOPWORDS and len(t) > 1]   # retire stop words
    if use_stemming:
        tokens = [_stemmer.stem(t) for t in tokens]   # racine des mots
    return " ".join(tokens).strip()
```
> La liste de stop words est **embarquée dans le code** au lieu d'être téléchargée : le
> projet fonctionne ainsi même sans connexion Internet.

## `featurize.py` — TF-IDF + split
```python
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label"], test_size=0.2,
    random_state=42, stratify=df["label"])   # split reproductible et stratifié

vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2),
                      min_df=2, sublinear_tf=True)
Xtr = vec.fit_transform(X_train)   # APPREND le vocabulaire sur le train
Xte = vec.transform(X_test)        # APPLIQUE le même vocabulaire au test
```
> **Attention.** Détail crucial : `fit_transform` sur le **train**, mais seulement `transform` sur le
> **test**. Sinon, le test « fuiterait » dans l'apprentissage (*data leakage*).

## `train.py` — l'entraînement
```python
cw = "balanced" if cfg["imbalance"] == "class_weight" else None
model = LogisticRegression(C=cfg["C"], class_weight=cw, max_iter=1000)
model.fit(Xtr, y_train)
joblib.dump(model, "models/model.joblib")   # sauvegarde du modèle
```

## `evaluate.py` — les métriques et figures
Calcule accuracy/precision/recall/F1/AUC, écrit `reports/metrics.json`, et génère la
**matrice de confusion** et la **courbe ROC** dans `reports/figures/`.

\newpage

# Suivi d'expériences avec MLflow

MLflow répond à la question : « quel réglage a donné quel résultat, et où est le modèle
correspondant ? » Il journalise chaque essai.

```bash
python src/train_mlflow.py
mlflow ui --backend-store-uri sqlite:///mlflow.db     #  http://localhost:5000
```

Dans l'interface web, chaque *run* (un modèle) montre ses **paramètres**, ses **métriques**
et son **artefact** (le modèle sauvegardé). Tu peux trier par F1, comparer deux runs côte à
côte, etc.

**» En clair.** : DVC assure la *reproductibilité* du pipeline ; MLflow assure la *traçabilité*
des expériences. Les deux sont complémentaires.

\newpage

# Déploiement — l'application Streamlit

## Lancer l'app
```bash
streamlit run app/streamlit_app.py       #  http://localhost:8501
```

## Ce qui se passe dans le code
1. **Chargement** (mis en cache) du vectoriseur et du modèle depuis `models/`.
2. L'utilisateur saisit un tweet.
3. On applique **exactement le même `clean_text`** qu'à l'entraînement (cohérence absolue).
4. `predict_proba` renvoie la probabilité ; au-dessus de 0,5  « SUSPECT ».
5. Chaque prédiction est **journalisée** dans `reports/prediction_log.csv` (pour le monitoring).

```python
cleaned = clean_text(txt)                     # même nettoyage qu'à l'entraînement
X = vec.transform([cleaned])
proba = model.predict_proba(X)[0, 1]          # probabilité classe « suspect »
pred = int(proba >= 0.5)
```
> **Attention.** Le modèle doit exister avant de lancer l'app : fais `dvc repro` ou `dvc pull` d'abord.

\newpage

# Monitoring et détection de dérive

Dans la barre latérale de l'app, une page **Monitoring** affiche :

- le **nombre total** de prédictions servies ;
- la **répartition** suspect / non suspect et l'histogramme des probabilités ;
- le **volume dans le temps** ;
- une **alerte de dérive**.

## Qu'est-ce que la dérive (drift) ?
En production, les tweets d'aujourd'hui ne ressemblent pas forcément à ceux de
l'entraînement (nouveaux sujets, nouvel argot, nouvelles arnaques). Si la distribution des
entrées s'éloigne trop de celle d'entraînement, le modèle se dégrade **silencieusement**.

La page compare deux indicateurs à leur valeur d'entraînement :

- la **longueur moyenne** des tweets entrants (référence ≈ 73,7 caractères) ;
- le **taux de « suspect »** prédit (référence ≈ 0,898).

Si l'écart dépasse un seuil (30 % sur la longueur, 20 points sur le taux), une **alerte rouge**
recommande un réentraînement.

**» En clair.** : le monitoring est le « tableau de bord santé » du modèle une fois en service.
Sans lui, on ne sait pas qu'un modèle est devenu mauvais avant qu'il ne cause des dégâts.

\newpage

# Déploiement cloud

Trois façons de mettre l'app en ligne (détails dans `DEPLOYMENT.md`).

## Docker — universel et reproductible
Le `Dockerfile` décrit une image contenant Python, les dépendances et le code.
```bash
docker build -t tweet-detector .        # construit l'image
docker run -p 8501:8501 tweet-detector  # lance le conteneur  http://localhost:8501
```
**» En clair.** : Docker emballe l'app *et son environnement* dans une « boîte » qui tourne
identiquement sur n'importe quelle machine.

## Streamlit Community Cloud — le plus simple (gratuit)
1. Pousser le dépôt sur GitHub.
2. https://share.streamlit.io  **New app**  choisir le dépôt et `app/streamlit_app.py`.

## Google Cloud Run — à partir du Docker
```bash
gcloud run deploy tweet-detector --source . --port 8501 --allow-unauthenticated
```

\newpage

# CI/CD avec GitHub Actions

**CI/CD** = Intégration / Déploiement Continus. À chaque `git push`, un robot (GitHub Actions)
vérifie automatiquement que le projet n'est pas cassé.

Le workflow `.github/workflows/ci.yml` exécute :

1. installation des dépendances,
2. **lint** (flake8) — repère les erreurs de syntaxe,
3. **tests** (`pytest`) — 8 tests unitaires et d'intégration,
4. vérification du pipeline DVC.

En local, avant de pousser :
```bash
pytest tests -v
```
*Sortie attendue* : `8 passed`.

**» En clair.** : la CI/CD est un filet de sécurité. Si un changement casse un test, tu le sais
immédiatement, avant que le code ne parte en production.

\newpage

# Dépannage (FAQ)

| Symptôme | Cause probable | Solution |
|----------|----------------|----------|
| `ModuleNotFoundError` | environnement non activé / dépendances manquantes | `source .venv/bin/activate` puis `pip install -r requirements.txt` |
| `streamlit: model not found` | modèle non généré | `dvc repro` ou `dvc pull` |
| `dvc pull` ne fait rien | données déjà présentes | c'est normal |
| Port 8501 occupé | une autre app tourne | `streamlit run app/streamlit_app.py --server.port 8502` |
| `git status` liste des fichiers modifiés sans raison | montage réseau (permissions) | disparaît sur disque local ; contenu identique au commit |
| MLflow : erreur file store | version récente de MLflow | utiliser `--backend-store-uri sqlite:///mlflow.db` |
| Téléchargement NLTK échoue | pas de réseau | inutile : la liste de stop words est embarquée |

\newpage

# Glossaire

- **Classification binaire** : prédire une étiquette parmi deux (ici 0 ou 1).
- **TF-IDF** : méthode qui transforme un texte en vecteur de nombres pondérés.
- **n-gramme** : séquence de *n* mots consécutifs (unigramme = 1 mot, bigramme = 2 mots).
- **Stop words** : mots très fréquents et peu informatifs (the, and, of…).
- **Stemming** : réduction d'un mot à sa racine (updating  updat).
- **Class weights** : pondération qui rend les erreurs sur la classe rare plus coûteuses.
- **SMOTE** : génération d'exemples synthétiques de la classe minoritaire.
- **Overfitting (sur-apprentissage)** : le modèle mémorise le train et généralise mal.
- **Precision / Recall / F1** : métriques d'évaluation d'un classifieur (voir §7.4).
- **ROC-AUC** : aire sous la courbe ROC ; mesure la qualité du classement (0,5 = hasard).
- **Validation croisée** : évaluation moyennée sur plusieurs découpages du train.
- **GridSearch** : recherche exhaustive du meilleur jeu d'hyperparamètres.
- **DVC** : outil de versionnement des données et d'orchestration de pipeline.
- **Pipeline** : suite d'étapes automatisées (preprocess  featurize  train  evaluate).
- **Remote (DVC)** : stockage distant des données/modèles (local, Drive, S3…).
- **MLflow** : outil de suivi d'expériences (params, métriques, modèles).
- **Data leakage (fuite)** : information du test qui contamine l'entraînement, gonflant les scores.
- **Drift (dérive)** : évolution des données en production qui dégrade le modèle.
- **CI/CD** : automatisation des tests/déploiements à chaque changement de code.

\newpage

# Checklist express

```bash
# 1. Environnement
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Données + pipeline
dvc pull            # récupère données et modèles
dvc repro           # reconstruit tout le pipeline
dvc metrics show    # affiche les métriques

# 3. Qualité
pytest tests -v     # 8 tests doivent passer

# 4. Suivi d'expériences (optionnel)
python src/train_mlflow.py
mlflow ui --backend-store-uri sqlite:///mlflow.db

# 5. Application + monitoring
streamlit run app/streamlit_app.py

# 6. Déploiement Docker (optionnel)
docker build -t tweet-detector . && docker run -p 8501:8501 tweet-detector
```

**Ordre de lecture conseillé pour l'oral** : §2 (organisation)  §3 (fil rouge)  §7
(concepts ML)  §6 (DVC)  §10 (monitoring). Bonne présentation !
