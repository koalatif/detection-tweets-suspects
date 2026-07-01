"""Partie 7 — Application Streamlit de detection de tweets suspects."""
import os, sys, joblib
import streamlit as st
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import clean_text

st.set_page_config(page_title="Detection de Tweets Suspects", page_icon="🐦", layout="centered")

@st.cache_resource
def load_artifacts():
    root = os.path.join(os.path.dirname(__file__), "..")
    vec = joblib.load(os.path.join(root, "models", "vectorizer.joblib"))
    model = joblib.load(os.path.join(root, "models", "model.joblib"))
    return vec, model

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

st.divider()
st.caption("Projet ML — Pipeline reproductible géré avec Git + DVC.")
