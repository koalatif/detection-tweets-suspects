"""Partie 1 — Prétraitement du texte.

Opérations de nettoyage appliquées à chaque tweet :
  - Conversion en minuscules
  - Suppression des URLs (http/https/www)
  - Suppression des mentions @user et des hashtags (#)
  - Suppression des caractères spéciaux / ponctuation / chiffres
  - Suppression des stop words (liste anglaise embarquée)
  - Stemming (Porter) — optionnel via params.yaml
"""
import re
import sys
import yaml
import pandas as pd
from nltk.stem import PorterStemmer

# Liste de stop words anglais embarquée (NLTK non téléchargeable hors-ligne).
STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because been
before being below between both but by can't cannot could couldn't did didn't do does
doesn't doing don't down during each few for from further had hadn't has hasn't have
haven't having he he'd he'll he's her here here's hers herself him himself his how how's
i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my
myself no nor not of off on once only or other ought our ours ourselves out over own same
shan't she she'd she'll she's should shouldn't so some such than that that's the their
theirs them themselves then there there's these they they'd they'll they're they've this
those through to too under until up very was wasn't we we'd we'll we're we've were weren't
what what's when when's where where's which while who who's whom why why's with won't would
wouldn't you you'd you'll you're you've your yours yourself yourselves u ur im
""".split())

_url_re = re.compile(r"http\S+|www\.\S+")
_mention_re = re.compile(r"@\w+")
_nonalpha_re = re.compile(r"[^a-z\s]")
_space_re = re.compile(r"\s+")
_stemmer = PorterStemmer()


def clean_text(text, use_stemming=True):
    """Nettoie un tweet et renvoie une chaîne normalisée."""
    text = str(text).lower()
    text = _url_re.sub(" ", text)
    text = _mention_re.sub(" ", text)
    text = text.replace("#", " ")
    text = _nonalpha_re.sub(" ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    if use_stemming:
        tokens = [_stemmer.stem(t) for t in tokens]
    return _space_re.sub(" ", " ".join(tokens)).strip()


def main(params_path="params.yaml"):
    params = yaml.safe_load(open(params_path))["preprocess"]
    df = pd.read_csv(params["input"])
    if params.get("drop_duplicates", True):
        df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)
    df["clean"] = df["message"].apply(lambda t: clean_text(t, params["use_stemming"]))
    df = df[df["clean"].str.len() > 0].reset_index(drop=True)
    out = params["output"]
    df[["clean", "label"]].to_csv(out, index=False)
    print(f"Pretraitement termine : {len(df)} lignes -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "params.yaml")
