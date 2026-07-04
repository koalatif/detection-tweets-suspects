"""Partie 7 — Application Streamlit de détection de tweets suspects."""
# isort: skip_file  (l'ordre des imports est volontaire : sys.path avant l'import local)
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import csv
import datetime
import joblib
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from preprocess import clean_text

ROOT = os.path.join(os.path.dirname(__file__), "..")
LOG = os.path.join(ROOT, "reports", "prediction_log.csv")


def _use_stemming():
    """Cohérence train/serve : même prétraitement qu'à l'entraînement.
    La source unique de vérité est params.yaml (clé preprocess.use_stemming)."""
    try:
        import yaml
        cfg = yaml.safe_load(open(os.path.join(ROOT, "params.yaml")))
        return bool(cfg["preprocess"]["use_stemming"])
    except Exception:
        return True


USE_STEMMING = _use_stemming()

st.set_page_config(
    page_title="Détection de Tweets Suspects",
    page_icon="",
    layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .title-gradient {
        background: -webkit-linear-gradient(45deg, #2563eb, #1e3a8a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 3rem;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    .result-card {
        padding: 30px;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1), 0 10px 10px -5px rgba(0,0,0,0.04);
        margin-bottom: 25px;
        transition: transform 0.3s ease;
    }
    .result-card:hover { transform: translateY(-5px); }
    .suspect-card { background: linear-gradient(135deg, #ef4444, #991b1b); }
    .normal-card { background: linear-gradient(135deg, #10b981, #047857); }
    .result-title { font-size: 1.5rem; font-weight: 700; letter-spacing: 2px; margin-bottom: 10px; opacity: 0.9; }
    .result-proba { font-size: 4rem; font-weight: 900; line-height: 1; margin-bottom: 10px; }
    .stTextArea textarea { border-radius: 15px; border: 2px solid #e2e8f0; padding: 15px; font-size: 1.1rem; }
    .stTextArea textarea:focus { border-color: #3b82f6; box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3); }
    .stButton>button {
        border-radius: 12px;
        width: 100%;
        font-weight: 800;
        font-size: 1.2rem;
        padding: 15px;
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.5);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb, #1e40af);
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.5);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    vec = joblib.load(os.path.join(ROOT, "models", "vectorizer.joblib"))
    model = joblib.load(os.path.join(ROOT, "models", "model.joblib"))
    vocab = vec.get_feature_names_out()
    coefs = model.coef_[0]
    return vec, model, vocab, coefs

@st.cache_resource
def load_bert_artifacts():
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        clf = joblib.load(os.path.join(ROOT, "models", "bert_head.joblib"))
        return encoder, clf
    except Exception:
        return None, None


def log_prediction(text, cleaned, proba, pred):
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "n_chars", "n_tokens_clean",
                        "proba_suspect", "prediction"])
        w.writerow([datetime.datetime.now().isoformat(timespec="seconds"),
                    len(text), len(cleaned.split()), round(proba, 4), pred])


vec, model, vocab, coefs = load_artifacts()

st.markdown(
    '<h1 class="title-gradient"> Détection de Tweets Suspects</h1>',
    unsafe_allow_html=True)
st.caption("Classification propulsée par l'IA (NLP & Régression Logistique), avec IA explicable en temps réel.")
st.markdown("<br>", unsafe_allow_html=True)

col_input, col_space, col_result = st.columns([1.2, 0.1, 1])

with col_input:
    st.markdown("###  Soumettre un message")
    model_choice = st.radio("Choisissez le moteur d'analyse :",
                            ["🤖 Classique (TF-IDF)", "🧠 Avancé (Transformers)"],
                            index=0, horizontal=True)
    txt = st.text_area("Saisissez le texte du tweet ci-dessous :",
                       "URGENT!!! Your bank account is locked. Verify now at http://secure-update-info.com",
                       height=180, label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button(" LANCER L'ANALYSE")
    st.markdown("<br>", unsafe_allow_html=True)

    if analyze_btn and txt.strip():
        with st.expander(" Voir les détails du pipeline NLP"):
            st.markdown("**Texte brut original :**\n> " + txt)
            if "Classique" in model_choice:
                cleaned = clean_text(txt, use_stemming=USE_STEMMING)
                st.markdown(
                    "**Texte après nettoyage, filtrage et tokenisation :**\n> " + (cleaned or "*(vide)*"))
            else:
                st.markdown("**Transformers :** Aucun nettoyage n'est appliqué car le modèle analyse la phrase originale avec son contexte complet.")

with col_result:
    if analyze_btn:
        if not txt.strip():
            st.warning("Veuillez saisir un texte à analyser.")
        else:
            is_bert = "Transformers" in model_choice
            
            if is_bert:
                encoder, bert_clf = load_bert_artifacts()
                if encoder is None:
                    st.error("Impossible de charger les Transformers. Vérifiez que `sentence-transformers` est installé.")
                    st.stop()
                X = encoder.encode([txt])
                proba = float(bert_clf.predict_proba(X)[0, 1])
                pred = int(proba >= 0.5)
                cleaned_log = txt
                can_explain = False
            else:
                cleaned = clean_text(txt, use_stemming=USE_STEMMING)
                if not cleaned:
                    st.info("Le texte ne contient aucun mot analysable après le filtre de vocabulaire.")
                    st.stop()
                X = vec.transform([cleaned])
                proba = float(model.predict_proba(X)[0, 1])
                pred = int(proba >= 0.5)
                cleaned_log = cleaned
                can_explain = True

            if pred == 1:
                st.markdown(f"""
                <div class="result-card suspect-card">
                    <div class="result-title">ALERTE SUSPECT</div>
                    <div class="result-proba">{proba * 100:.1f} %</div>
                    <div style="font-size: 1.1rem; opacity: 0.9;">Probabilité d'anomalie ou de phishing</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card normal-card">
                    <div class="result-title">TWEET SAIN</div>
                    <div class="result-proba">{(1 - proba) * 100:.1f} %</div>
                    <div style="font-size: 1.1rem; opacity: 0.9;">Probabilité d'authenticité</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("###  Raisonnement de l'IA")
            
            if not can_explain:
                st.info("💡 **Modèle Avancé (Transformers)** : Ce modèle analyse le contexte global de la phrase grâce à un réseau de neurones. Contrairement au modèle classique, le poids de la décision est réparti sur tout le contexte, il n'est donc pas possible d'isoler l'impact individuel de chaque mot.")
            else:
                st.caption("Top des mots ayant influencé cette décision (contribution algorithmique) :")

                cx = X.tocoo()
                contributions = []
                for j, v in zip(cx.col, cx.data):
                    word = vocab[j]
                    weight = coefs[j] * v          # influence = tfidf × coefficient
                    contributions.append((word, weight))

                contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                top_contributions = contributions[:8]

                if not top_contributions:
                    st.info("Le modèle n'a reconnu aucun mot décisif par rapport à son apprentissage.")
                else:
                    df_c = pd.DataFrame(top_contributions, columns=["Mot", "Poids"])
                    df_c["Impact"] = [
                        "Suspect (Anomalie)" if x > 0 else "Normal (Sain)" for x in df_c["Poids"]]
                    chart = alt.Chart(df_c).mark_bar(cornerRadiusEnd=4, size=20).encode(
                        x=alt.X("Poids:Q", title="Contribution algorithmique"),
                        y=alt.Y("Mot:N", sort="-x", title=""),
                        color=alt.Color("Impact:N", scale=alt.Scale(
                            domain=["Suspect (Anomalie)", "Normal (Sain)"],
                            range=["#ef4444", "#10b981"]), legend=None),
                        tooltip=["Mot", "Poids"]
                    ).properties(height=len(df_c) * 30 + 50)
                    st.altair_chart(chart, use_container_width=True)

            log_prediction(txt, cleaned_log, proba, pred)

st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("Projet ML MLOps — page **Monitoring** (barre latérale) pour les statistiques des prédictions. ")
