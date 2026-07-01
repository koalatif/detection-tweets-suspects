"""Bonus — Dashboard de monitoring des predictions et de derive (data drift)."""
import os, sys
import streamlit as st
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
LOG = os.path.join(ROOT, "reports", "prediction_log.csv")

st.set_page_config(page_title="Monitoring", page_icon="📊", layout="wide")
st.title("📊 Dashboard de Monitoring")
st.caption("Suivi des prédictions servies et détection de dérive (drift)")

# Reference : longueur moyenne des tweets a l'entrainement (issue de l'EDA)
REF_MEAN_CHARS = 73.7        # moyenne caracteres sur le train
REF_SUSPECT_RATE = 0.898     # part de la classe suspecte a l'entrainement

if not os.path.exists(LOG):
    st.info("Aucune prédiction enregistrée pour l'instant. "
            "Analysez des tweets dans la page principale pour alimenter ce tableau de bord.")
    st.stop()

df = pd.read_csv(LOG)
df["timestamp"] = pd.to_datetime(df["timestamp"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Prédictions totales", len(df))
suspect_rate = df["prediction"].mean()
c2.metric("Taux de tweets suspects", f"{suspect_rate*100:.1f} %",
          delta=f"{(suspect_rate-REF_SUSPECT_RATE)*100:+.1f} pts vs train")
c3.metric("Probabilité moyenne", f"{df['proba_suspect'].mean()*100:.1f} %")
mean_chars = df["n_chars"].mean()
c4.metric("Longueur moyenne (car.)", f"{mean_chars:.0f}",
          delta=f"{mean_chars-REF_MEAN_CHARS:+.0f} vs train")

st.divider()
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribution des prédictions")
    counts = df["prediction"].value_counts().rename({0: "Non suspect", 1: "Suspect"})
    st.bar_chart(counts)

with col2:
    st.subheader("Distribution des probabilités")
    hist = np.histogram(df["proba_suspect"], bins=10, range=(0, 1))[0]
    st.bar_chart(pd.DataFrame({"proba": np.round(np.linspace(0.05, 0.95, 10), 2),
                               "count": hist}).set_index("proba"))

st.subheader("Volume de prédictions dans le temps")
ts = df.set_index("timestamp").resample("1min").size()
st.line_chart(ts)

# --- Alerte de derive ---
st.divider()
st.subheader("🚦 Détection de dérive (data drift)")
drift_chars = abs(mean_chars - REF_MEAN_CHARS) / REF_MEAN_CHARS
drift_rate = abs(suspect_rate - REF_SUSPECT_RATE)
if drift_chars > 0.30 or drift_rate > 0.20:
    st.error(f"⚠️ Dérive détectée — longueur: {drift_chars*100:.0f}% d'écart, "
             f"taux suspect: {drift_rate*100:.0f} pts d'écart. Un réentraînement est conseillé.")
else:
    st.success("✅ Aucune dérive significative détectée par rapport aux données d'entraînement.")

with st.expander("Journal brut des prédictions"):
    st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)
