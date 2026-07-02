"""Partie 1 — Analyse exploratoire des donnees. Genere les figures dans reports/figures/."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")  # backend headless, avant tout import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
from preprocess import clean_text

sns.set_theme(style="whitegrid")
FIG = "reports/figures"

def main():
    df = pd.read_csv("data/raw/tweets_suspect.csv")
    print("Shape:", df.shape)
    print("Valeurs manquantes:\n", df.isnull().sum())
    print("Doublons:", df.duplicated().sum())
    print("Distribution des classes:\n", df["label"].value_counts())

    labels = {0: "Non suspect (0)", 1: "Suspect (1)"}

    # 1. Distribution des classes
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["label"].value_counts().sort_index()
    bars = ax.bar([labels[i] for i in counts.index], counts.values,
                  color=["#4C72B0", "#C44E52"])
    ax.set_title("Distribution des classes")
    ax.set_ylabel("Nombre de tweets")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x()+b.get_width()/2, v+400, f"{v}\n({v/len(df)*100:.1f}%)",
                ha="center", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{FIG}/class_distribution.png", dpi=120); plt.close()

    # 2. Distribution de la longueur des tweets
    df["len"] = df["message"].str.len()
    fig, ax = plt.subplots(figsize=(7, 4))
    for lab, c in [(0, "#4C72B0"), (1, "#C44E52")]:
        sns.histplot(df[df.label==lab]["len"], bins=40, color=c, alpha=0.55,
                     label=labels[lab], ax=ax, stat="density")
    ax.set_title("Longueur des tweets par classe"); ax.set_xlabel("Nombre de caracteres")
    ax.legend(); plt.tight_layout(); plt.savefig(f"{FIG}/length_distribution.png", dpi=120); plt.close()

    # 3. Nombre de mots par classe (boxplot)
    df["nwords"] = df["message"].str.split().apply(len)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="label", y="nwords", hue="label", palette=["#4C72B0","#C44E52"],
                legend=False, ax=ax)
    ax.set_xticklabels([labels[0], labels[1]])
    ax.set_title("Nombre de mots par tweet"); ax.set_xlabel(""); ax.set_ylabel("Mots")
    plt.tight_layout(); plt.savefig(f"{FIG}/words_boxplot.png", dpi=120); plt.close()

    # 4. Top mots + wordclouds (sur texte nettoye)
    df["clean"] = df["message"].apply(lambda t: clean_text(t, True))
    for lab in [0, 1]:
        text = " ".join(df[df.label==lab]["clean"])
        wc = WordCloud(width=800, height=400, background_color="white",
                       colormap="Blues" if lab==0 else "Reds").generate(text)
        wc.to_file(f"{FIG}/wordcloud_class{lab}.png")

    # 5. Top 15 mots frequents globaux
    all_words = Counter(" ".join(df["clean"]).split()).most_common(15)
    fig, ax = plt.subplots(figsize=(7, 4))
    words, freqs = zip(*all_words)
    sns.barplot(x=list(freqs), y=list(words), hue=list(words), palette="viridis",
                legend=False, ax=ax)
    ax.set_title("Top 15 des mots (apres nettoyage)"); ax.set_xlabel("Frequence")
    plt.tight_layout(); plt.savefig(f"{FIG}/top_words.png", dpi=120); plt.close()

    print("Figures EDA generees dans", FIG)

if __name__ == "__main__":
    main()
