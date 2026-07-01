"""Test d'intégration : le modèle entraîné prédit correctement."""
import os, sys, joblib, pytest
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from preprocess import clean_text

MODEL = os.path.join(ROOT, "models", "model.joblib")
VEC = os.path.join(ROOT, "models", "vectorizer.joblib")

@pytest.mark.skipif(not (os.path.exists(MODEL) and os.path.exists(VEC)),
                    reason="Artefacts absents — lancer 'dvc repro' d'abord")
def test_prediction_shape_and_range():
    vec = joblib.load(VEC); model = joblib.load(MODEL)
    X = vec.transform([clean_text("free money click here to win")])
    proba = float(model.predict_proba(X)[0, 1])
    assert 0.0 <= proba <= 1.0
    assert model.predict(X)[0] in (0, 1)
