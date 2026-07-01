"""Partie 7 — Application Streamlit de detection de tweets suspects."""
import os, sys, joblib, csv, datetime
import streamlit as st
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import clean_text

st.set_page_config(page_title="Detection de Tweets Suspects", page_icon="🐦", layout="centered")
ROOT = os.path.join(os.path.dirname(__file__), "..")
LOG = os.path.join(ROOT, "reports", "prediction_log.csv")

@st.cache_resource
def load_artifacts():
    vec = joblib.load(os.path.join(ROOT, "models", "vectorizer.joblib"))
    model = joblib.load(os.path.join(ROOT, "models", "model.joblib"))
    return vec, model

def log_prediction(text, cleaned, proba, pred):
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "n_chars", "n_tokens_clean", "proba_suspect", "prediction"])
        w.writerow([datetime.datetime.now().isoformat(timespec="seconds"),
                    len(text), len(cleaned.split()), round(proba, 4), pred])

vec, model = load_artifacts()

st.title("🐦 Détection de Tweets Suspects")
st.caption("Classification automatique par TF-IDF + Régression Logistique (class weights)")

txt = st.text_area("Saisissez un tweet à analyser :",
                   "Win a FREE iPhone now!!! Click http://bit.ly/xyz", height=120)

if st.button("Analyser", type="primary"):
    if not txt.strip():
        st.warning("Veuillez saisir un texte.")
    else:
        cleaned = clean_text(txt, use_stemming=True)
        X = vec.transform([cleaned])
        proba = float(model.predict_proba(X)[0, 1])
        pred = int(proba >= 0.5)
        label = "SUSPECT ⚠️" if pred == 1 else "NON SUSPECT ✅"
        st.subheader(f"Résultat : {label}")
        st.metric("Probabilité (classe suspecte)", f"{proba*100:.1f} %")
        st.progress(proba)
        with st.expander("Détails du prétraitement"):
            st.write("**Texte nettoyé :**", cleaned or "*(vide après nettoyage)*")
        log_prediction(txt, cleaned, proba, pred)

st.divider()
st.caption("Projet ML — Pipeline reproductible géré avec Git + DVC. "
           "Voir la page **Monitoring** dans la barre latérale 📊")
