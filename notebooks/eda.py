"""
Exploratory Data Analysis — Heart Disease UCI Dataset
Generates all required EDA visualizations: histograms, heatmap, class balance, etc.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart_disease_cleaned.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "screenshots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
})
PALETTE = ["#3B82F6", "#EF4444"]
sns.set_theme(style="whitegrid", palette=PALETTE)

FEATURE_LABELS = {
    "age": "Age (years)", "sex": "Sex", "cp": "Chest Pain Type",
    "trestbps": "Resting BP (mmHg)", "chol": "Cholesterol (mg/dl)",
    "fbs": "Fasting Blood Sugar", "restecg": "Resting ECG",
    "thalach": "Max Heart Rate", "exang": "Exercise Angina",
    "oldpeak": "ST Depression", "slope": "ST Slope",
    "ca": "Major Vessels", "thal": "Thal",
}

df = pd.read_csv(DATA_PATH)
df["target_label"] = df["target"].map({0: "No Disease", 1: "Disease"})

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


# ────────────────────────────────────────────────────────────────
# 1. Class Balance
# ────────────────────────────────────────────────────────────────
def plot_class_balance():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    counts = df["target_label"].value_counts()

    # Bar chart
    bars = axes[0].bar(counts.index, counts.values, color=PALETTE, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     str(val), ha="center", va="bottom", fontsize=12, fontweight="bold")
    axes[0].set_title("Class Distribution", fontsize=14, fontweight="bold", pad=10)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_ylim(0, max(counts) * 1.15)

    # Pie chart
    axes[1].pie(counts.values, labels=counts.index, colors=PALETTE,
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2},
                textprops={"fontsize": 12})
    axes[1].set_title("Class Proportions", fontsize=14, fontweight="bold", pad=10)

    fig.suptitle("Heart Disease Class Balance", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "1_class_balance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ────────────────────────────────────────────────────────────────
# 2. Numeric Feature Histograms
# ────────────────────────────────────────────────────────────────
def plot_histograms():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, col in enumerate(NUMERIC):
        for label, color in zip(["No Disease", "Disease"], PALETTE):
            subset = df[df["target_label"] == label][col]
            axes[i].hist(subset, bins=20, alpha=0.65, color=color, label=label, edgecolor="white")
        axes[i].set_title(FEATURE_LABELS[col], fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Value", fontsize=10)
        axes[i].set_ylabel("Count", fontsize=10)
        axes[i].legend(fontsize=9)

    # Summary stats text in last cell
    stats = df[NUMERIC].describe().round(1)
    axes[5].axis("off")
    text = "Summary Statistics (Numeric Features)\n\n"
    for col in NUMERIC:
        text += f"{FEATURE_LABELS[col][:20]:20s}: μ={stats.loc['mean', col]:.1f}, σ={stats.loc['std', col]:.1f}\n"
    axes[5].text(0.05, 0.95, text, transform=axes[5].transAxes, fontsize=9,
                 verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="#EEF2FF", alpha=0.8))

    fig.suptitle("Numeric Feature Distributions by Class", fontsize=15, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "2_histograms.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ────────────────────────────────────────────────────────────────
# 3. Correlation Heatmap
# ────────────────────────────────────────────────────────────────
def plot_correlation_heatmap():
    corr = df.drop(columns=["target_label"]).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, ax=ax,
        cbar_kws={"shrink": 0.7, "label": "Pearson r"},
        annot_kws={"size": 8}, linewidths=0.5, square=True
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "3_correlation_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ────────────────────────────────────────────────────────────────
# 4. Box Plots — Numeric vs Target
# ────────────────────────────────────────────────────────────────
def plot_boxplots():
    fig, axes = plt.subplots(1, 5, figsize=(16, 5))
    for i, col in enumerate(NUMERIC):
        sns.boxplot(
            x="target_label", y=col, data=df,
            palette=PALETTE, ax=axes[i],
            width=0.5, flierprops={"marker": "o", "markersize": 3}
        )
        axes[i].set_title(FEATURE_LABELS[col], fontsize=11, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        axes[i].tick_params(axis="x", rotation=20, labelsize=9)

    fig.suptitle("Numeric Features vs Heart Disease", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "4_boxplots.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ────────────────────────────────────────────────────────────────
# 5. Categorical Feature Counts
# ────────────────────────────────────────────────────────────────
def plot_categorical_counts():
    cats = ["sex", "cp", "exang", "slope", "ca", "thal"]
    cat_labels = {
        "sex": {0: "Female", 1: "Male"},
        "cp": {0: "Asymptomatic", 1: "Atypical Angina", 2: "Non-anginal", 3: "Typical Angina"},
        "exang": {0: "No", 1: "Yes"},
        "slope": {0: "Upsloping", 1: "Flat", 2: "Downsloping"},
        "ca": {i: str(i) for i in range(4)},
        "thal": {1: "Normal", 2: "Fixed Defect", 3: "Reversable"},
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, col in enumerate(cats):
        cross = df.groupby([col, "target_label"]).size().unstack(fill_value=0)
        cross.index = [cat_labels[col].get(idx, str(idx)) for idx in cross.index]
        cross.plot(kind="bar", ax=axes[i], color=PALETTE, edgecolor="white",
                   linewidth=1, width=0.6, legend=(i == 0))
        axes[i].set_title(FEATURE_LABELS[col], fontsize=12, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=30, labelsize=8)
        axes[i].set_ylabel("Count", fontsize=9)
        if i == 0:
            axes[i].legend(fontsize=9)

    fig.suptitle("Categorical Features vs Heart Disease", fontsize=15, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "5_categorical_features.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ────────────────────────────────────────────────────────────────
# 6. Age vs Max HR Scatter (key clinical insight)
# ────────────────────────────────────────────────────────────────
def plot_age_thalach_scatter():
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, color in zip(["No Disease", "Disease"], PALETTE):
        subset = df[df["target_label"] == label]
        ax.scatter(subset["age"], subset["thalach"],
                   c=color, label=label, alpha=0.65, s=50, edgecolors="white", linewidth=0.5)

    # Trend lines
    for label, color in zip(["No Disease", "Disease"], PALETTE):
        subset = df[df["target_label"] == label]
        z = np.polyfit(subset["age"], subset["thalach"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(subset["age"].min(), subset["age"].max(), 100)
        ax.plot(x_line, p(x_line), color=color, linewidth=2, linestyle="--", alpha=0.8)

    ax.set_xlabel("Age (years)", fontsize=12)
    ax.set_ylabel("Max Heart Rate Achieved", fontsize=12)
    ax.set_title("Age vs Max Heart Rate by Disease Status", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "6_age_vs_thalach.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ────────────────────────────────────────────────────────────────
# 7. Missing Value Analysis
# ────────────────────────────────────────────────────────────────
def plot_missing_values():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    missing = df.drop(columns=["target_label"]).isnull().sum()
    axes[0].barh(missing.index, missing.values,
                 color=["#EF4444" if v > 0 else "#3B82F6" for v in missing.values])
    axes[0].set_title("Missing Values per Feature", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Count")
    axes[0].axvline(0, color="gray", linewidth=0.8)
    for i, v in enumerate(missing.values):
        axes[0].text(v + 0.1, i, str(v), va="center", fontsize=9)

    # Data types
    dtypes = df.drop(columns=["target_label"]).dtypes.value_counts()
    axes[1].pie(dtypes.values, labels=[str(t) for t in dtypes.index],
                autopct="%1.0f%%", colors=["#3B82F6", "#F59E0B"],
                wedgeprops={"edgecolor": "white", "linewidth": 2},
                textprops={"fontsize": 11})
    axes[1].set_title("Feature Data Types", fontsize=12, fontweight="bold")

    fig.suptitle("Data Quality Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "7_missing_values.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    print(f"\nDataset: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"Target balance: {df['target'].value_counts().to_dict()}")
    print(f"\nGenerating EDA plots → {PLOTS_DIR}\n")

    plot_class_balance()
    plot_histograms()
    plot_correlation_heatmap()
    plot_boxplots()
    plot_categorical_counts()
    plot_age_thalach_scatter()
    plot_missing_values()

    print("\n✅ All EDA plots generated successfully!")
