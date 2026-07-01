"""Partie 6 — GridSearch (hyperparametres) + Partie 4 — comparaison des strategies de desequilibre."""
import json, joblib, scipy.sparse as sp, numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score
from imblearn.over_sampling import SMOTE

Xtr = sp.load_npz("data/processed/X_train.npz")
Xte = sp.load_npz("data/processed/X_test.npz")
y_train, y_test = joblib.load("data/processed/y.joblib")

# --- GridSearch sur Logistic Regression ---
grid = {"C": [0.1, 1, 10, 100], "penalty": ["l2"], "class_weight": ["balanced"]}
gs = GridSearchCV(LogisticRegression(max_iter=1000, n_jobs=-1), grid,
                  scoring="f1", cv=StratifiedKFold(3, shuffle=True, random_state=42), n_jobs=-1)
gs.fit(Xtr, y_train)
best = {"best_params": gs.best_params_, "best_cv_f1": round(gs.best_score_, 4)}
print("GridSearch:", best)

# --- Comparaison des strategies de desequilibre (memes hyperparams) ---
strategies = {}
base = dict(C=gs.best_params_["C"], max_iter=1000, n_jobs=-1)
for name in ["none", "class_weight", "smote"]:
    if name == "none":
        m = LogisticRegression(**base); Xt, yt = Xtr, y_train
    elif name == "class_weight":
        m = LogisticRegression(class_weight="balanced", **base); Xt, yt = Xtr, y_train
    else:
        Xt, yt = SMOTE(random_state=42).fit_resample(Xtr, y_train)
        m = LogisticRegression(**base)
    m.fit(Xt, yt)
    yp = m.predict(Xte)
    strategies[name] = {
        "precision": round(precision_score(y_test, yp), 4),
        "recall": round(recall_score(y_test, yp), 4),
        "f1": round(f1_score(y_test, yp), 4),
        # recall sur la classe minoritaire (0) = capacite a detecter le "non suspect"
        "recall_class0": round(recall_score(y_test, yp, pos_label=0), 4),
    }
    print(name, strategies[name])

json.dump({"gridsearch": best, "imbalance_strategies": strategies},
          open("reports/optimization.json", "w"), indent=2)
print("-> reports/optimization.json")
