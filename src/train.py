"""Partie 4 — Entrainement du modele retenu pour le pipeline DVC."""
import sys, yaml, joblib, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

def build_model(cfg):
    cw = "balanced" if cfg["imbalance"] == "class_weight" else None
    if cfg["model"] == "logistic":
        return LogisticRegression(C=cfg.get("C", 1.0), class_weight=cw,
                                  max_iter=1000, n_jobs=-1)
    raise ValueError(cfg["model"])

def main(params_path="params.yaml"):
    cfg = yaml.safe_load(open(params_path))["train"]
    Xtr = sp.load_npz("data/processed/X_train.npz")
    y_train, _ = joblib.load("data/processed/y.joblib")
    if cfg["imbalance"] == "smote":
        from imblearn.over_sampling import SMOTE
        Xtr, y_train = SMOTE(random_state=42).fit_resample(Xtr, y_train)
    model = build_model(cfg)
    model.fit(Xtr, y_train)
    joblib.dump(model, "models/model.joblib")
    print(f"Modele entraine ({cfg['model']}, imbalance={cfg['imbalance']}) -> models/model.joblib")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "params.yaml")
