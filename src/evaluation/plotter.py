import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

OUT = Path("results/visualizations")
OUT.mkdir(parents=True, exist_ok=True)

def plot_medal_trends(df, top_n=10):
    top    = df.groupby("Country")["TotalMedals"].sum().nlargest(top_n).index.tolist()
    subset = df[df["Country"].isin(top)]
    fig, ax = plt.subplots(figsize=(13, 5))
    for country, grp in subset.groupby("Country"):
        ax.plot(grp["Year"], grp["TotalMedals"], marker="o", label=country)
    ax.set_title(f"Top {top_n} countries — medal trends")
    ax.set_xlabel("Year"); ax.set_ylabel("Total medals")
    ax.legend(bbox_to_anchor=(1.01, 1), fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT / "medal_trends.png", dpi=150); plt.close()

def plot_feature_importance(model, feature_names, title="feature_importance"):
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:15]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feature_names[i] for i in idx[::-1]], imp[idx[::-1]])
    ax.set_title(title.replace("_", " ").title())
    plt.tight_layout()
    fig.savefig(OUT / f"{title}.png", dpi=150); plt.close()

def plot_actual_vs_pred(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=20)
    lim = max(float(max(y_true)), float(max(y_pred))) * 1.05
    ax.plot([0, lim], [0, lim], "r--")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    plt.tight_layout()
    fig.savefig(OUT / "actual_vs_pred.png", dpi=150); plt.close()

def plot_cluster_heatmap(df, feat_cols):
    pivot = df.groupby("Cluster")[feat_cols].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(pivot.T, annot=True, fmt=".1f", ax=ax, cmap="YlOrRd")
    ax.set_title("Cluster profiles")
    plt.tight_layout()
    fig.savefig(OUT / "cluster_heatmap.png", dpi=150); plt.close()

def plot_elbow(k_scores: dict):
    ks          = sorted(k_scores.keys())
    inertias    = [k_scores[k]["inertia"]    for k in ks]
    silhouettes = [k_scores[k]["silhouette"] for k in ks]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(ks, inertias,    marker="o"); ax1.set_title("Elbow curve")
    ax2.plot(ks, silhouettes, marker="o", color="green")
    ax2.set_title("Silhouette scores")
    plt.tight_layout()
    fig.savefig(OUT / "elbow_silhouette.png", dpi=150); plt.close()