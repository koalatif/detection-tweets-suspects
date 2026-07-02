"""Bonus — Représentation avancée par Transformers (Sentence-BERT).

Ce script encode les tweets avec un modèle *Sentence-Transformers* pré-entraîné
(`all-MiniLM-L6-v2`, une variante distillée de BERT), puis entraîne un classifieur
léger (Régression Logistique) sur ces embeddings sémantiques de dimension 384.

Il constitue une alternative « avancée » à la représentation TF-IDF : au lieu de compter
des mots, BERT capture le *sens* contextuel des phrases.

Prérequis :
    pip install sentence-transformers torch

Usage :
    python src/train_bert.py                # sur tout le jeu nettoyé
    python src/train_bert.py --sample 10000 # échantillon (plus rapide pour tester)

Sorties :
    reports/bert_metrics.json               # métriques du modèle BERT
    reports/figures/confusion_matrix_bert.png
    models/bert_head.joblib                 # classifieur entraîné sur les embeddings

NB : le premier lancement télécharge le modèle (~90 Mo) depuis HuggingFace ; une
connexion Internet est donc requise une seule fois (ensuite il est mis en cache).
"""
import os
import sys
import json
import argparse
import joblib
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0,
                    help="Taille d'échantillon (0 = tout le jeu de données)")
    ap.add_argument("--model", default="all-MiniLM-L6-v2",
                    help="Nom du modèle Sentence-Transformers")
    args = ap.parse_args()

    # Imports lourds à l'intérieur pour un message clair si non installés / hors-ligne
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("Manquant : pip install sentence-transformers torch")

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, confusion_matrix,
                                 ConfusionMatrixDisplay)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Données nettoyées (réutilise la sortie du pipeline)
    df = pd.read_csv("data/processed/clean.csv").dropna(subset=["clean"])
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)
    print(f"Encodage de {len(df)} tweets avec « {args.model} »…")

    # 2. Chargement du modèle Transformer et encodage
    try:
        encoder = SentenceTransformer(args.model)
    except Exception as e:
        sys.exit(f"Impossible de charger le modèle (connexion Internet requise "
                 f"au premier lancement) : {e}")
    X = encoder.encode(df["clean"].tolist(), batch_size=64,
                       show_progress_bar=True, convert_to_numpy=True)
    y = df["label"].values

    # 3. Split + classifieur léger sur les embeddings
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(C=10, class_weight="balanced", max_iter=1000, n_jobs=-1)
    clf.fit(X_tr, y_tr)

    # 4. Évaluation
    y_pred = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)[:, 1]
    metrics = {
        "representation": f"sentence-transformers/{args.model}",
        "embedding_dim": int(X.shape[1]),
        "n_samples": int(len(df)),
        "accuracy": round(accuracy_score(y_te, y_pred), 4),
        "precision": round(precision_score(y_te, y_pred), 4),
        "recall": round(recall_score(y_te, y_pred), 4),
        "f1": round(f1_score(y_te, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_te, y_proba), 4),
    }
    os.makedirs("reports/figures", exist_ok=True)
    json.dump(metrics, open("reports/bert_metrics.json", "w"), indent=2)
    joblib.dump(clf, "models/bert_head.joblib")
    print(json.dumps(metrics, indent=2))

    # 5. Matrice de confusion
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Non susp.", "Suspect"]).plot(
        ax=ax, cmap="Purples", colorbar=False)
    ax.set_title(f"Matrice de confusion — {args.model}")
    plt.tight_layout()
    plt.savefig("reports/figures/confusion_matrix_bert.png", dpi=120)
    print("-> reports/bert_metrics.json, models/bert_head.joblib, figure générée.")


if __name__ == "__main__":
    main()
