import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
data = sorted(json.load(open("reports/model_comparison.json")), key=lambda d: -d["f1"])
models = [d["model"] for d in data]
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]
x = np.arange(len(models)); w = 0.15
fig, ax = plt.subplots(figsize=(11, 5.5))
for i, met in enumerate(metrics):
    ax.bar(x + i*w, [d[met] for d in data], w, label=met.upper(), color=colors[i])
ax.set_xticks(x + 2*w); ax.set_xticklabels(models, rotation=15)
ax.set_ylim(0.90, 1.0); ax.set_ylabel("Score")
ax.set_title("Comparaison des modeles (jeu de test)")
ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.22))
plt.tight_layout(); plt.savefig("reports/figures/model_comparison.png", dpi=120); plt.close()

# Figure strategies de desequilibre
opt = json.load(open("reports/optimization.json"))["imbalance_strategies"]
strats = list(opt.keys()); f1s = [opt[s]["f1"] for s in strats]; rc0 = [opt[s]["recall_class0"] for s in strats]
x = np.arange(len(strats))
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(x-0.2, f1s, 0.4, label="F1 (classe 1)", color="#4C72B0")
ax.bar(x+0.2, rc0, 0.4, label="Recall classe 0 (minoritaire)", color="#C44E52")
ax.set_xticks(x); ax.set_xticklabels(["Aucune","Class weights","SMOTE"])
ax.set_ylim(0.7, 1.0); ax.set_title("Effet des strategies de gestion du desequilibre")
ax.legend(); 
for i,(a,b) in enumerate(zip(f1s,rc0)):
    ax.text(i-0.2,a+0.005,f"{a:.3f}",ha="center",fontsize=8)
    ax.text(i+0.2,b+0.005,f"{b:.3f}",ha="center",fontsize=8)
plt.tight_layout(); plt.savefig("reports/figures/imbalance_comparison.png", dpi=120); plt.close()
print("Figures de comparaison generees.")
