"""Parties 4-5-6 — Comparaison de plusieurs modeles avec gestion du desequilibre.

Usage: python src/compare_models.py <model_name>
Chaque appel entraine 1 modele, calcule les metriques test + une CV 3-fold (F1),
et ajoute la ligne au fichier reports/model_comparison.json (append).
"""
import sys, json, time, os, joblib, numpy as np, scipy.sparse as sp
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

RESULTS = "reports/model_comparison.json"

def get_model(name, spw):
    if name == "LogisticRegression":
        return LogisticRegression(C=10, class_weight="balanced", max_iter=1000, n_jobs=-1)
    if name == "MultinomialNB":
        return MultinomialNB(alpha=0.1)          # NB gere mal class_weight -> alpha
    if name == "LinearSVC":
        return CalibratedClassifierCV(LinearSVC(C=1, class_weight="balanced"), cv=3)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=120, max_depth=40, n_jobs=-1,
                                      class_weight="balanced", random_state=42)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.2,
                             scale_pos_weight=spw, tree_method="hist",
                             n_jobs=-1, eval_metric="logloss", random_state=42)
    raise ValueError(name)

def main(name):
    Xtr = sp.load_npz("data/processed/X_train.npz")
    Xte = sp.load_npz("data/processed/X_test.npz")
    y_train, y_test = joblib.load("data/processed/y.joblib")
    spw = (y_train == 0).sum() / (y_train == 1).sum()  # pour XGBoost (ratio inverse)
    # note: classe minoritaire = 0, on veut ponderer -> scale_pos_weight base sur pos=1
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    t0 = time.time()
    model = get_model(name, spw)
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    try:
        y_proba = model.predict_proba(Xte)[:, 1]
    except Exception:
        y_proba = model.decision_function(Xte)
    # CV F1 (3-fold) sur le train
    cv = cross_val_score(get_model(name, spw), Xtr, y_train,
                         cv=StratifiedKFold(3, shuffle=True, random_state=42),
                         scoring="f1", n_jobs=-1)
    row = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "cv_f1_mean": round(cv.mean(), 4),
        "cv_f1_std": round(cv.std(), 4),
        "train_time_s": round(time.time() - t0, 1),
    }
    data = json.load(open(RESULTS)) if os.path.exists(RESULTS) else []
    data = [d for d in data if d["model"] != name] + [row]
    json.dump(data, open(RESULTS, "w"), indent=2)
    print(json.dumps(row, indent=2))

if __name__ == "__main__":
    main(sys.argv[1])
