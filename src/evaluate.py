"""Partie 6 — Evaluation : metriques, matrice de confusion, courbe ROC/AUC."""
import sys, json, yaml, joblib, scipy.sparse as sp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay)

def main(params_path="params.yaml"):
    p = yaml.safe_load(open(params_path))
    Xte = sp.load_npz("data/processed/X_test.npz")
    _, y_test = joblib.load("data/processed/y.joblib")
    model = joblib.load("models/model.joblib")
    y_pred = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    json.dump(metrics, open(p["evaluate"]["metrics_file"], "w"), indent=2)
    print(json.dumps(metrics, indent=2))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Non susp.", "Suspect"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matrice de confusion (modele final)")
    plt.tight_layout(); plt.savefig("reports/figures/confusion_matrix.png", dpi=120); plt.close()

    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#C44E52", lw=2, label=f"AUC = {metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("Taux de faux positifs"); ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbe ROC (modele final)"); ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig("reports/figures/roc_curve.png", dpi=120); plt.close()
    print("Figures d'evaluation generees.")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "params.yaml")
