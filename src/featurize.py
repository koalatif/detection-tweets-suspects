"""Parties 3 & 5 — Representation TF-IDF + split train/test."""
import sys, yaml, joblib, scipy.sparse as sp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def main(params_path="params.yaml"):
    p = yaml.safe_load(open(params_path))
    fp, spx = p["featurize"], p["split"]
    df = pd.read_csv(p["preprocess"]["output"]).dropna(subset=["clean"])
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"].values, df["label"].values,
        test_size=spx["test_size"], random_state=spx["random_state"],
        stratify=df["label"].values)

    vec = TfidfVectorizer(max_features=fp["max_features"],
                          ngram_range=(1, fp["ngram_max"]), min_df=fp["min_df"],
                          sublinear_tf=True)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    joblib.dump(vec, "models/vectorizer.joblib")
    sp.save_npz("data/processed/X_train.npz", Xtr)
    sp.save_npz("data/processed/X_test.npz", Xte)
    joblib.dump((y_train, y_test), "data/processed/y.joblib")
    print(f"TF-IDF: {Xtr.shape[1]} features | train {Xtr.shape[0]} | test {Xte.shape[0]}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "params.yaml")
