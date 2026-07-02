"""Partie 3 (optionnel) — Réduction de dimension t-SNE de l'espace TF-IDF.

Réduit les vecteurs TF-IDF (creux, 20 000 dimensions) en 2D pour visualiser la
séparabilité des classes. Pipeline : TruncatedSVD (50 comp.) -> t-SNE (2D).
Un échantillon stratifié est utilisé car t-SNE est coûteux.

Usage : python src/tsne_viz.py [n_echantillon]
Sortie : reports/figures/tsne_tfidf.png
"""
import sys
import joblib
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

sns.set_theme(style="whitegrid")


def main(n=3500):
    # Réutilise la matrice TF-IDF du train produite par featurize.py
    X = sp.load_npz("data/processed/X_train.npz")
    y_train, _ = joblib.load("data/processed/y.joblib")
    y = np.asarray(y_train)

    # Échantillon stratifié (garde la proportion des classes)
    rng = np.random.RandomState(42)
    idx0 = rng.choice(np.where(y == 0)[0], size=int(n * (y == 0).mean()), replace=False)
    idx1 = rng.choice(np.where(y == 1)[0], size=n - len(idx0), replace=False)
    idx = np.concatenate([idx0, idx1]); rng.shuffle(idx)
    Xs, ys = X[idx], y[idx]

    # 1) Pré-réduction SVD (rapide, adaptée aux matrices creuses)
    svd = TruncatedSVD(n_components=50, random_state=42)
    Xr = svd.fit_transform(Xs)
    var = svd.explained_variance_ratio_.sum()
    print(f"SVD 50 comp. — variance expliquée : {var:.1%}")

    # 2) t-SNE en 2D
    ts = TSNE(n_components=2, perplexity=30, init="pca",
              learning_rate="auto", random_state=42)
    emb = ts.fit_transform(Xr)

    # 3) Tracé
    fig, ax = plt.subplots(figsize=(9, 7))
    for lab, col, name in [(0, "#4C72B0", "Non suspect (0)"), (1, "#C44E52", "Suspect (1)")]:
        m = ys == lab
        ax.scatter(emb[m, 0], emb[m, 1], s=14, c=col, alpha=0.5,
                   edgecolors="white", linewidths=0.2, label=name)
    ax.set_title("Visualisation t-SNE des Tweets (TF-IDF)", fontsize=14)
    ax.legend(markerscale=1.6, fontsize=11)
    plt.tight_layout()
    plt.savefig("reports/figures/tsne_tfidf.png", dpi=120)
    print("-> reports/figures/tsne_tfidf.png")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3500)
