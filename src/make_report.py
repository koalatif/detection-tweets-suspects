"""Génère le rapport PDF du projet (reports/Rapport_Detection_Tweets_Suspects.pdf).

Reprend les métriques réelles (reports/*.json) et toutes les figures de reports/figures/.
Usage : python src/make_report.py
"""
import json
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak)

R = "."
FIG = R + "/reports/figures"
comp = json.load(open(R+"/reports/model_comparison.json"))
opt = json.load(open(R+"/reports/optimization.json"))

styles = getSampleStyleSheet()
NAVY = colors.HexColor("#1F3A5F"); RED = colors.HexColor("#C44E52")
h1 = ParagraphStyle("h1", parent=styles["Heading1"], textColor=NAVY, spaceBefore=14, spaceAfter=8, fontSize=16)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=RED, spaceBefore=10, spaceAfter=6, fontSize=12.5)
body = ParagraphStyle("body", parent=styles["BodyText"], alignment=TA_JUSTIFY, fontSize=10.2, leading=15, spaceAfter=6)
cap = ParagraphStyle("cap", parent=styles["BodyText"], alignment=TA_CENTER, fontSize=8.5, textColor=colors.grey, spaceAfter=12)
title = ParagraphStyle("title", parent=styles["Title"], textColor=NAVY, fontSize=24, leading=28)
sub = ParagraphStyle("sub", parent=styles["Normal"], alignment=TA_CENTER, fontSize=12, textColor=colors.grey)

story = []
def P(t, s=body): story.append(Paragraph(t, s))
def SP(h=0.3): story.append(Spacer(1, h*cm))
def IMG(name, w=15, capt=None):
    story.append(Image(f"{FIG}/{name}", width=w*cm, height=w*cm*0.5))
    if capt: story.append(Paragraph(capt, cap))

# ---- Page de titre ----
SP(5)
P("Détection Automatique de<br/>Tweets Suspects", title)
SP(0.4)
P("Projet de Machine Learning — Cycle de vie complet<br/>Exploration · DVC · Modélisation · Déploiement", sub)
SP(1.2)
story.append(Table([[""]], colWidths=[6*cm], style=TableStyle([("LINEABOVE",(0,0),(-1,0),1.2,RED)])))
SP(0.6)
P("Rapport technique — Classification de texte (NLP)", sub)
SP(6)
P("Pipeline reproductible géré avec <b>Git + DVC</b> · Application <b>Streamlit</b>", sub)
story.append(PageBreak())

# ---- 1. Introduction ----
P("1. Introduction", h1)
P("Les réseaux sociaux constituent une source majeure d'information mais peuvent aussi véhiculer "
  "des contenus suspects, offensants ou trompeurs. Ce projet développe une <b>solution complète de "
  "classification automatique</b> permettant d'identifier les tweets suspects, en couvrant l'ensemble "
  "du cycle de vie d'un projet de Machine Learning : de l'exploration des données jusqu'au déploiement, "
  "avec versionnement des données et des modèles via <b>DVC (Data Version Control)</b>.")
P("Le jeu de données comprend <b>60 000 tweets</b> annotés par une étiquette binaire : "
  "<b>1 = suspect</b>, <b>0 = non suspect</b>. Après suppression des doublons, "
  "<b>59 450 tweets</b> sont conservés. Le problème est un cas de <b>classification binaire de texte</b> "
  "fortement déséquilibré.")

# ---- 2. Exploration ----
P("2. Analyse exploratoire des données", h1)
P("Les données comportent deux variables (<i>message</i>, <i>label</i>), sans valeur manquante et avec "
  "293 doublons exacts. La caractéristique majeure du jeu de données est un <b>fort déséquilibre des "
  "classes</b> : la classe « suspect » représente environ <b>89,8 %</b> des observations contre "
  "<b>10,2 %</b> pour la classe « non suspect ». Ce déséquilibre oriente tous les choix méthodologiques "
  "ultérieurs (rééquilibrage, métriques d'évaluation).")
IMG("class_distribution.png", 11, "Figure 1 — Distribution des classes (déséquilibre ~90/10).")
P("La longueur des tweets (médiane ≈ 69 caractères) et le nombre de mots se distribuent de façon "
  "similaire entre les deux classes, ce qui indique que la longueur n'est pas un discriminant fort et "
  "que le signal utile réside dans le <b>vocabulaire</b>.")
IMG("length_distribution.png", 12, "Figure 2 — Distribution de la longueur des tweets par classe.")
P("Le diagramme en boîte du nombre de mots par tweet confirme ce constat : les deux classes présentent "
  "des médianes et des dispersions proches, sans écart marqué exploitable directement.")
IMG("words_boxplot.png", 10, "Figure 3 — Nombre de mots par tweet, par classe (boîtes à moustaches).")
P("Les nuages de mots réalisés sur le texte nettoyé, présentés ci-dessous pour les <b>deux classes</b>, "
  "révèlent un vocabulaire largement partagé (mots courants du langage quotidien). Cette forte similarité "
  "annonce une frontière de décision subtile plutôt que trivialement séparable — un point important repris "
  "en discussion.")
story.append(Table([[Image(f"{FIG}/wordcloud_class0.png", width=8*cm, height=4*cm),
                     Image(f"{FIG}/wordcloud_class1.png", width=8*cm, height=4*cm)]],
                   colWidths=[8.2*cm, 8.2*cm]))
story.append(Paragraph("Figure 4 — Nuages de mots : classe « non suspect » (gauche) et « suspect » (droite).", cap))
P("Enfin, le classement des mots les plus fréquents (après nettoyage) donne un aperçu du lexique dominant "
  "du corpus.")
IMG("top_words.png", 12, "Figure 5 — Mots les plus fréquents après nettoyage.")

# ---- 3. Prétraitement ----
P("3. Prétraitement du texte", h1)
P("Le nettoyage (module <font face='Courier'>src/preprocess.py</font>) applique successivement : "
  "conversion en <b>minuscules</b> ; suppression des <b>URLs</b>, des <b>mentions</b> "
  "<font face='Courier'>@user</font> et des <b>hashtags</b> ; suppression de la <b>ponctuation, des "
  "chiffres et caractères spéciaux</b> ; retrait des <b>stop words</b> anglais ; et enfin "
  "<b>stemming de Porter</b> pour ramener les mots à leur racine.")
P("<b>Justification.</b> Les URLs et mentions sont propres à chaque tweet et n'apportent pas de signal "
  "généralisable ; les stop words diluent l'information ; le stemming réduit la dimensionnalité en "
  "regroupant les variantes morphologiques (<i>update / updating / updated → updat</i>), ce qui améliore "
  "la densité du signal pour les modèles linéaires.")

# ---- 4. Représentation ----
P("4. Représentation des données", h1)
P("Les tweets nettoyés sont transformés par <b>TF-IDF</b> (Term Frequency – Inverse Document Frequency), "
  "avec <b>uni-grammes et bi-grammes</b>, un maximum de <b>20 000 features</b>, une fréquence documentaire "
  "minimale <font face='Courier'>min_df=2</font> et une pondération <font face='Courier'>sublinear_tf</font>. "
  "TF-IDF est retenu car il est performant, interprétable et particulièrement adapté aux textes courts et "
  "creux comme les tweets, tout en restant peu coûteux comparé aux embeddings profonds (BERT). Les "
  "bi-grammes capturent des expressions (<i>« free iphone »</i>) utiles à la détection. Une variante "
  "avancée par <b>Sentence-BERT</b> est également fournie (<font face='Courier'>src/train_bert.py</font>).")
P("<b>Réduction de dimension (t-SNE).</b> Pour visualiser l'organisation de l'espace TF-IDF, "
  "les vecteurs (20 000 dimensions) sont projetés en 2D via une pré-réduction <b>TruncatedSVD</b> "
  "(50 composantes) suivie de <b>t-SNE</b>, sur un échantillon stratifié de 3 500 tweets. "
  "La projection fait apparaître de nombreux micro-clusters (regroupements par thèmes/vocabulaire), "
  "mais <b>les deux classes y sont entremêlées</b> sans frontière nette : les points « suspect » et "
  "« non suspect » se mélangent dans les mêmes amas. Cela corrobore visuellement le constat de "
  "l'analyse exploratoire — une séparation subtile plutôt que linéairement évidente.")
IMG("tsne_tfidf.png", 13, "Figure 6 — Projection t-SNE de l'espace TF-IDF (échantillon de 3 500 tweets).")

# ---- 5. Gestion du déséquilibre ----
P("5. Gestion du déséquilibre des classes", h1)
P("Trois stratégies ont été comparées à hyperparamètres égaux : aucune correction, <b>class weights</b> "
  "(pondération inverse des fréquences) et <b>SMOTE</b> (sur-échantillonnage synthétique). Le rappel de "
  "la <b>classe minoritaire (0)</b> est l'indicateur clé, car c'est la capacité à ne pas « rater » les "
  "tweets non suspects noyés dans la majorité.")
im = opt["imbalance_strategies"]
rows = [["Stratégie","Precision","Recall (cl.1)","F1","Recall classe 0"]]
for k,lab in [("none","Aucune"),("class_weight","Class weights"),("smote","SMOTE")]:
    d=im[k]; rows.append([lab,f"{d['precision']:.3f}",f"{d['recall']:.3f}",f"{d['f1']:.3f}",f"{d['recall_class0']:.3f}"])
t = Table(rows, colWidths=[3.2*cm,2.6*cm,2.8*cm,2*cm,3*cm])
t.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),colors.white),
    ("FONTSIZE",(0,0),(-1,-1),9),("ALIGN",(1,0),(-1,-1),"CENTER"),
    ("GRID",(0,0),(-1,-1),0.4,colors.grey),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#EEF2F7")]),
    ("BACKGROUND",(0,2),(-1,2),colors.HexColor("#D6EAD6")),("FONTNAME",(0,2),(-1,2),"Helvetica-Bold")]))
story.append(t); story.append(Paragraph("Tableau 1 — Effet des stratégies de rééquilibrage.", cap))
P("Les <b>class weights</b> offrent le meilleur compromis : le meilleur F1 (0,988) <i>et</i> le meilleur "
  "rappel de la classe minoritaire (0,873, contre 0,779 sans correction). SMOTE dégrade ici le F1 en "
  "générant du bruit dans un espace TF-IDF très creux. <b>Les class weights sont donc retenus.</b>")
IMG("imbalance_comparison.png", 11, "Figure 7 — Comparaison des stratégies de rééquilibrage.")
story.append(PageBreak())

# ---- 6. Comparaison des modèles ----
P("6. Construction et comparaison des modèles", h1)
P("Cinq algorithmes ont été évalués sur le jeu de test (20 %) après séparation stratifiée, avec une "
  "<b>validation croisée 3-fold</b> (F1) sur le train. Toutes les métriques minimales requises sont "
  "rapportées.")
rows = [["Modèle","Accuracy","Precision","Recall","F1","ROC-AUC","CV F1"]]
best = max(comp, key=lambda d:d["f1"])
for d in sorted(comp, key=lambda x:-x["f1"]):
    rows.append([d["model"],f"{d['accuracy']:.3f}",f"{d['precision']:.3f}",f"{d['recall']:.3f}",
                 f"{d['f1']:.3f}",f"{d['roc_auc']:.3f}",f"{d['cv_f1_mean']:.3f}"])
st=[("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),colors.white),
    ("FONTSIZE",(0,0),(-1,-1),8.5),("ALIGN",(1,0),(-1,-1),"CENTER"),
    ("GRID",(0,0),(-1,-1),0.4,colors.grey),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#EEF2F7")])]
story.append(Table(rows, colWidths=[3.7*cm,2*cm,2*cm,1.8*cm,1.6*cm,1.9*cm,1.7*cm], style=TableStyle(st)))
story.append(Paragraph("Tableau 2 — Comparaison des modèles (jeu de test + CV 3-fold).", cap))
P("Les modèles linéaires (LinearSVC, Régression Logistique) et XGBoost obtiennent des performances "
  "quasi identiques (F1 ≈ 0,988). Naive Bayes maximise le rappel mais au prix de la précision. Random "
  "Forest est légèrement en retrait. Nous retenons la <b>Régression Logistique</b> : performance de "
  "premier plan, probabilités calibrées natives (utiles au déploiement), interprétabilité et coût "
  "d'entraînement minimal (2,3 s contre 22,6 s pour XGBoost).")
IMG("model_comparison.png", 15, "Figure 8 — Comparaison visuelle des cinq modèles.")

# ---- 7. Optimisation ----
P("7. Optimisation des hyperparamètres", h1)
bp = opt["gridsearch"]["best_params"]
P(f"Une recherche par grille (<b>GridSearchCV</b>, 3-fold, scoring F1) sur le paramètre de régularisation "
  f"<font face='Courier'>C ∈ {{0.1, 1, 10, 100}}</font> a été menée sur la Régression Logistique. "
  f"Les paramètres optimaux obtenus sont : <b>C = {bp['C']}</b>, <b>penalty = {bp['penalty']}</b>, "
  f"<b>class_weight = {bp['class_weight']}</b> (F1 CV = {opt['gridsearch']['best_cv_f1']:.3f}).")

# ---- 8. Évaluation finale ----
P("8. Évaluation du modèle final", h1)
mf = json.load(open(R+"/reports/metrics.json"))
P(f"Le modèle final (TF-IDF + Régression Logistique, class weights, C=10) atteint sur le jeu de test : "
  f"<b>Accuracy {mf['accuracy']:.3f}</b>, <b>Precision {mf['precision']:.3f}</b>, "
  f"<b>Recall {mf['recall']:.3f}</b>, <b>F1 {mf['f1']:.3f}</b> et <b>ROC-AUC {mf['roc_auc']:.3f}</b>.")
story.append(Table([[Image(f"{FIG}/confusion_matrix.png", width=8*cm, height=6.4*cm),
                     Image(f"{FIG}/roc_curve.png", width=8*cm, height=6.4*cm)]],
                   colWidths=[8.2*cm,8.2*cm]))
story.append(Paragraph("Figure 9 — Matrice de confusion et courbe ROC (AUC = 0,953) du modèle final.", cap))

# ---- 9. Déploiement ----
P("9. Déploiement", h1)
P("Le modèle est déployé via une <b>application Streamlit</b> (<font face='Courier'>app/streamlit_app.py</font>) "
  "qui permet de saisir un tweet, d'appliquer le même prétraitement, puis d'afficher la <b>prédiction</b> "
  "(suspect / non suspect) et la <b>probabilité</b> associée. Le vectoriseur et le modèle sont chargés "
  "depuis <font face='Courier'>models/</font>, artefacts versionnés par DVC.")

# ---- 10. Gestion DVC ----
P("10. Gestion des données et reproductibilité (DVC)", h1)
P("Le projet est intégralement versionné avec <b>Git + DVC</b>. Le dataset et les artefacts (vectoriseur, "
  "modèle, jeux transformés) sont suivis par DVC avec un <b>remote</b> de stockage configuré. Le pipeline "
  "déclaratif (<font face='Courier'>dvc.yaml</font>) enchaîne quatre étapes — "
  "<b>preprocess → featurize → train → evaluate</b> — paramétrées par "
  "<font face='Courier'>params.yaml</font>. N'importe qui peut reproduire l'intégralité des résultats "
  "avec :")
P("<font face='Courier'>dvc pull &nbsp;&amp;&amp;&nbsp; dvc repro &nbsp;&amp;&amp;&nbsp; dvc metrics show</font>",
  ParagraphStyle("code", parent=body, backColor=colors.HexColor("#EEF2F7"), borderPadding=6, alignment=TA_CENTER))
P("Grâce au suivi des dépendances, la modification d'un paramètre ne réexécute que les étapes impactées, "
  "garantissant efficacité et <b>reproductibilité totale</b>.")

# ---- 11. Discussion ----
P("11. Discussion", h1)
P("<b>Limites.</b> L'analyse exploratoire (Figures 3 et 4) montre que les deux classes partagent un "
  "vocabulaire très proche : la frontière sémantique entre « suspect » et « non suspect » est faible dans "
  "ce jeu de données, et les métriques élevées sont en partie portées par le fort a priori majoritaire "
  "(89,8 %). C'est précisément pourquoi nous privilégions le <b>F1, l'AUC (0,953) et le rappel de la classe "
  "minoritaire</b> plutôt que la seule accuracy : l'AUC de 0,953 confirme l'existence d'un signal réel, "
  "au-delà de la simple prédiction majoritaire.")
P("<b>Difficultés rencontrées.</b> Le déséquilibre des classes ; le choix d'une stratégie de "
  "rééquilibrage adaptée à un espace TF-IDF creux (où SMOTE s'avère contre-productif) ; et "
  "l'industrialisation d'un pipeline reproductible avec DVC.")
P("<b>Perspectives.</b> Explorer des représentations contextuelles (<b>Sentence-Transformers, BERT</b>) "
  "susceptibles de mieux capter la sémantique ; enrichir le nettoyage (gestion des emojis, "
  "lemmatisation) ; ajouter le suivi d'expériences avec <b>MLflow</b> et une <b>CI/CD</b> ; et déployer "
  "l'application sur le cloud avec un tableau de bord de monitoring.")

# ---- 12. Conclusion ----
P("12. Conclusion", h1)
P("Ce projet met en œuvre l'intégralité du cycle de vie d'une solution de Machine Learning pour la "
  "détection de tweets suspects : exploration et prétraitement du texte, représentation TF-IDF, gestion "
  "du déséquilibre, comparaison de cinq modèles, optimisation par GridSearch, évaluation complète "
  "(F1 = 0,988 ; AUC = 0,953), déploiement Streamlit et pipeline entièrement reproductible avec Git et "
  "DVC. L'ensemble constitue une base robuste et industrialisable.")

# ---- 13. Extensions bonus ----
P("13. Extensions bonus", h1)
P("Plusieurs extensions ont été mises en place au-delà du périmètre demandé :", body)
P("<b>MLflow</b> — suivi d'expériences en complément de DVC : le script "
  "<font face='Courier'>src/train_mlflow.py</font> journalise paramètres, métriques et modèles "
  "des cinq algorithmes (backend SQLite), visualisables via <font face='Courier'>mlflow ui</font>.", body)
P("<b>CI/CD (GitHub Actions)</b> — le workflow <font face='Courier'>.github/workflows/ci.yml</font> "
  "installe les dépendances, exécute le lint (flake8) et la suite de tests <font face='Courier'>pytest</font> "
  "(8 tests unitaires et d'intégration), et vérifie le pipeline DVC à chaque push.", body)
P("<b>Déploiement cloud</b> — un <font face='Courier'>Dockerfile</font>, une configuration Streamlit et "
  "un guide <font face='Courier'>DEPLOYMENT.md</font> permettent un déploiement sur Streamlit Community "
  "Cloud, Hugging Face Spaces ou tout hébergeur Docker (Cloud Run, Render, Fly.io).", body)
P("<b>Dashboard de monitoring</b> — une seconde page Streamlit suit en temps réel le volume de "
  "prédictions, la distribution des classes et des probabilités, et signale toute <b>dérive (drift)</b> "
  "par rapport aux statistiques d'entraînement.", body)
P("<b>Transformers avancés (Sentence-BERT)</b> — le script <font face='Courier'>src/train_bert.py</font> "
  "encode les tweets avec un modèle <font face='Courier'>all-MiniLM-L6-v2</font> et entraîne un "
  "classifieur sur ces embeddings sémantiques, en alternative à TF-IDF.", body)

doc = SimpleDocTemplate(R+"/reports/Rapport_Detection_Tweets_Suspects.pdf", pagesize=A4,
    topMargin=1.6*cm, bottomMargin=1.6*cm, leftMargin=2*cm, rightMargin=2*cm,
    title="Détection de Tweets Suspects", author="Projet ML")

def footer(canvas, doc):
    canvas.saveState(); canvas.setFont("Helvetica",8); canvas.setFillColor(colors.grey)
    canvas.drawString(2*cm,1*cm,"Détection de Tweets Suspects — Rapport technique")
    canvas.drawRightString(19*cm,1*cm,f"Page {doc.page}"); canvas.restoreState()

doc.build(story, onFirstPage=lambda c,d:None, onLaterPages=footer)
print("PDF genere avec les figures EDA completes.")
