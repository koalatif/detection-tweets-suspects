"""Bonus — Suivi d'experiences avec MLflow (en complement de DVC).

Entraine les 5 modeles, journalise parametres, metriques et artefacts dans MLflow.
Usage:  python src/train_mlflow.py
Visualiser: mlflow ui  (puis http://localhost:5000)
"""
import joblib, scipy.sparse as sp, mlflow, mlflow.sklearn
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score)
from compare_models import get_model, RESULTS
import json, os

MODELS = ["LogisticRegression", "MultinomialNB", "LinearSVC", "RandomForest", "XGBoost"]
PARAMS = {
    "LogisticRegression": {"C": 10, "class_weight": "balanced"},
    "MultinomialNB": {"alpha": 0.1},
    "LinearSVC": {"C": 1, "class_weight": "balanced"},
    "RandomForest": {"n_estimators": 120, "max_depth": 40, "class_weight": "balanced"},
    "XGBoost": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.2},
}

def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("detection-tweets-suspects")
    Xtr = sp.load_npz("data/processed/X_train.npz")
    Xte = sp.load_npz("data/processed/X_test.npz")
    y_train, y_test = joblib.load("data/processed/y.joblib")
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    for name in MODELS:
        with mlflow.start_run(run_name=name):
            model = get_model(name, spw)
            model.fit(Xtr, y_train)
            yp = model.predict(Xte)
            try: proba = model.predict_proba(Xte)[:, 1]
            except Exception: proba = model.decision_function(Xte)
            metrics = {
                "accuracy": accuracy_score(y_test, yp),
                "precision": precision_score(y_test, yp),
                "recall": recall_score(y_test, yp),
                "f1": f1_score(y_test, yp),
                "roc_auc": roc_auc_score(y_test, proba),
            }
            mlflow.log_param("model", name)
            mlflow.log_param("vectorizer", "tfidf_20000_bigram")
            for k, v in PARAMS[name].items():
                mlflow.log_param(k, v)
            mlflow.log_metrics(metrics)
            try:
                mlflow.sklearn.log_model(model, name="model")
            except Exception as e:
                print("  (log_model ignore:", e, ")")
            print(f"{name:20s} F1={metrics['f1']:.4f} AUC={metrics['roc_auc']:.4f}")
    print("\nExperiences journalisees dans mlflow.db  ->  lancer: mlflow ui --backend-store-uri sqlite:///mlflow.db")

if __name__ == "__main__":
    main()
