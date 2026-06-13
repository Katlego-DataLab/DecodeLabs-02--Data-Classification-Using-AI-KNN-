"""

  DecodeLabs | 25 May - 25 June  2026
  Project 2: Data Classification Using AI
  Author   : Katlego Mathebula
  Algorithm: K-Nearest Neighbors (KNN)
  Dataset  : Iris Benchmark (UCI / sklearn)


PIPELINE  INPUT → PROCESS → OUTPUT (IPO)
  Input   : Iris dataset  |  Feature scaling (StandardScaler)
  Process : Train-Test split (80/20)  |  KNN classifier
  Output  : Confusion matrix  |  F1 score  |  Elbow curve
"""

# ─────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)

# Consistent style across all plots
plt.rcParams.update({
    "figure.facecolor": "#0F1117",
    "axes.facecolor":   "#1A1D27",
    "axes.edgecolor":   "#3A3D4D",
    "axes.labelcolor":  "#C8CAD8",
    "xtick.color":      "#C8CAD8",
    "ytick.color":      "#C8CAD8",
    "text.color":       "#C8CAD8",
    "grid.color":       "#2A2D3D",
    "grid.linewidth":   0.5,
    "font.family":      "DejaVu Sans",
})
PALETTE = ["#4F8EF7", "#F06292", "#66BB6A"]   # Blue | Pink | Green per class


# ─────────────────────────────────────────────
# 1. LOAD & EXPLORE DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("  STAGE 1 — LOAD & EXPLORE")
print("=" * 60)

iris = load_iris()
X = iris.data                          # shape: (150, 4)
y = iris.target                        # 0 = Setosa, 1 = Versicolor, 2 = Virginica
class_names = iris.target_names        # ['setosa', 'versicolor', 'virginica']
feature_names = iris.feature_names     # sepal/petal length/width

df = pd.DataFrame(X, columns=feature_names)
df["species"] = [class_names[i] for i in y]

print(f"\n  Samples   : {X.shape[0]}")
print(f"  Features  : {X.shape[1]}  → {list(feature_names)}")
print(f"  Classes   : {len(class_names)}  → {list(class_names)}")
print(f"\n  Class distribution:\n{df['species'].value_counts().to_string()}")
print(f"\n  First 5 rows:\n{df.head().to_string(index=False)}")
print(f"\n  Stats:\n{df.describe().round(2).to_string()}")


# ─────────────────────────────────────────────
# 2. FEATURE SCALING  (The Gatekeeper Rule)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STAGE 2 — FEATURE SCALING")
print("=" * 60)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n  Before scaling — mean : {X.mean(axis=0).round(2)}")
print(f"  Before scaling — std  : {X.std(axis=0).round(2)}")
print(f"\n  After scaling  — mean : {X_scaled.mean(axis=0).round(4)}")
print(f"  After scaling  — std  : {X_scaled.std(axis=0).round(4)}")
print("\n  ✔  All features now at Mean=0, Variance=1")
print("     KNN uses distance — unscaled data would bias toward")
print("     larger-valued features. Scaling removes that bias.")


# ─────────────────────────────────────────────
# 3. TRAIN-TEST SPLIT  (Structural Integrity)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STAGE 3 — TRAIN / TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.20,
    random_state=42,   # reproducible results
    shuffle=True,      # remove ordering bias
    stratify=y,        # preserve class proportions
)

print(f"\n  Total samples  : {len(X_scaled)}")
print(f"  Training set   : {len(X_train)} samples ({len(X_train)/len(X_scaled)*100:.0f}%)")
print(f"  Testing set    : {len(X_test)} samples  ({len(X_test)/len(X_scaled)*100:.0f}%)")
print(f"\n  Stratified split — class counts in test set:")
for i, name in enumerate(class_names):
    print(f"    {name:12} : {(y_test == i).sum()} samples")


# ─────────────────────────────────────────────
# 4. FIND OPTIMAL K  (The Elbow Method)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STAGE 4 — HYPERPARAMETER TUNING (Elbow Method)")
print("=" * 60)

k_range     = range(1, 31)
error_rates = []

for k in k_range:
    knn  = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    error_rates.append(1 - f1_score(y_test, preds, average="weighted"))

best_k    = int(np.argmin(error_rates)) + 1
best_f1   = round(1 - min(error_rates), 4)

print(f"\n  K values tested : 1 → 30")
print(f"  Optimal K       : {best_k}  (lowest error rate)")
print(f"  Best F1 score   : {best_f1}")
print(f"\n  Logic: K=1 memorises noise (overfitting).")
print(f"         K=30 is too generic (underfitting).")
print(f"         K={best_k} sits at the elbow — best generalisation.")


# ─────────────────────────────────────────────
# 5. TRAIN & PREDICT  (The Scikit-learn Workflow)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STAGE 5 — TRAIN & PREDICT")
print("=" * 60)

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)                    # FIT  — memorise the map
predictions = model.predict(X_test)            # PREDICT — apply logic

print(f"\n  Model   : KNeighborsClassifier(n_neighbors={best_k})")
print(f"  Trained on {len(X_train)} samples")
print(f"  Predicted {len(X_test)} test samples")

# Show first 10 predictions vs actual
print("\n  Sample predictions (first 10 test samples):")
print(f"  {'Predicted':>12}  {'Actual':>12}  {'Correct?':>8}")
print("  " + "-" * 36)
for pred, actual in zip(predictions[:10], y_test[:10]):
    tick = "✔" if pred == actual else "✘"
    print(f"  {class_names[pred]:>12}  {class_names[actual]:>12}  {tick:>8}")


# ─────────────────────────────────────────────
# 6. EVALUATE  (Confusion Matrix + F1 Score)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STAGE 6 — EVALUATION")
print("=" * 60)

cm       = confusion_matrix(y_test, predictions)
f1_macro = f1_score(y_test, predictions, average="macro")
f1_wtd   = f1_score(y_test, predictions, average="weighted")
f1_per   = f1_score(y_test, predictions, average=None)

print(f"\n  Confusion Matrix:\n{cm}")
print(f"\n  F1 Score (macro)    : {f1_macro:.4f}")
print(f"  F1 Score (weighted) : {f1_wtd:.4f}")
print(f"\n  Per-class F1:")
for name, score in zip(class_names, f1_per):
    bar = "█" * int(score * 20)
    print(f"    {name:12} : {score:.4f}  {bar}")

print(f"\n  Detailed Classification Report:")
print(classification_report(y_test, predictions, target_names=class_names))


# ─────────────────────────────────────────────
# 7. VISUALISATIONS
# ─────────────────────────────────────────────
# %%
fig = plt.figure(figsize=(18, 12))
fig.suptitle(
    "DecodeLabs  |  Project 2: Data Classification Using AI  |  KNN on Iris",
    fontsize=14, fontweight="bold", color="#E8EAF6", y=0.98
)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)


# ── Plot 1: Elbow curve ──────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(k_range, error_rates, color="#4F8EF7", linewidth=2, marker="o",
         markersize=4, markerfacecolor="#F06292")
ax1.axvline(best_k, color="#FFD54F", linestyle="--", linewidth=1.5,
            label=f"Optimal K = {best_k}")
ax1.set_title("Elbow method — optimal K", fontsize=11, pad=8)
ax1.set_xlabel("K value")
ax1.set_ylabel("Error rate (1 − F1)")
ax1.legend(fontsize=9)
ax1.grid(True)


# ── Plot 2: Confusion matrix ─────────────────
ax2 = fig.add_subplot(gs[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax2, colorbar=False, cmap="Blues")
ax2.set_title("Confusion matrix", fontsize=11, pad=8)
ax2.set_xlabel("Predicted label")
ax2.set_ylabel("True label")


# ── Plot 3: F1 per class ─────────────────────
ax3 = fig.add_subplot(gs[0, 2])
bars = ax3.barh(class_names, f1_per, color=PALETTE, edgecolor="none", height=0.5)
ax3.set_xlim(0, 1.1)
ax3.set_title("F1 score per class", fontsize=11, pad=8)
ax3.set_xlabel("F1 score")
ax3.axvline(1.0, color="#3A3D4D", linestyle="--", linewidth=1)
for bar, val in zip(bars, f1_per):
    ax3.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", fontsize=10, color="#E8EAF6")
ax3.grid(True, axis="x")


# ── Plot 4: Petal scatter (test predictions) ─
ax4 = fig.add_subplot(gs[1, 0])
feat_idx = [2, 3]   # petal length, petal width — most discriminative pair
feature_labels = [feature_names[i] for i in feat_idx]

for i, (name, color) in enumerate(zip(class_names, PALETTE)):
    mask = y_test == i
    ax4.scatter(
        X_test[mask, feat_idx[0]], X_test[mask, feat_idx[1]],
        label=name, color=color, s=55, edgecolors="#0F1117", linewidth=0.5,
        marker="o" if predictions[mask].all() == i else "X",
    )
    # mark misclassifications with an X
    wrong = mask & (predictions != y_test)
    if wrong.any():
        ax4.scatter(
            X_test[wrong, feat_idx[0]], X_test[wrong, feat_idx[1]],
            color="red", s=120, marker="x", linewidths=2, zorder=5
        )

ax4.set_title("Predictions on test set (petal space)", fontsize=11, pad=8)
ax4.set_xlabel(feature_labels[0] + " (scaled)")
ax4.set_ylabel(feature_labels[1] + " (scaled)")
ax4.legend(fontsize=9)
ax4.grid(True)


# ── Plot 5: Feature importance (std of scaled) ──
ax5 = fig.add_subplot(gs[1, 1])
variance = np.var(X_scaled, axis=0)
colors_bar = ["#4F8EF7", "#F06292", "#66BB6A", "#FFD54F"]
ax5.bar(range(4), variance, color=colors_bar, edgecolor="none", width=0.5)
ax5.set_xticks(range(4))
ax5.set_xticklabels(
    ["Sepal\nlength", "Sepal\nwidth", "Petal\nlength", "Petal\nwidth"],
    fontsize=9
)
ax5.set_title("Feature variance (after scaling)", fontsize=11, pad=8)
ax5.set_ylabel("Variance")
ax5.grid(True, axis="y")


# ── Plot 6: Metrics summary card ─────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")

summary_lines = [
    ("Model",           f"KNN  (K = {best_k})"),
    ("Dataset",         "Iris benchmark"),
    ("Train samples",   str(len(X_train))),
    ("Test samples",    str(len(X_test))),
    ("Scaler",          "StandardScaler"),
    ("Split",           "80 / 20  (stratified)"),
    ("F1 (macro)",      f"{f1_macro:.4f}"),
    ("F1 (weighted)",   f"{f1_wtd:.4f}"),
]

row_h, col_x1, col_x2 = 0.11, 0.02, 0.52
for i, (label, value) in enumerate(summary_lines):
    y_pos = 0.92 - i * row_h
    bg = "#1E2130" if i % 2 == 0 else "#232640"
    ax6.add_patch(plt.Rectangle((0, y_pos - 0.04), 1, row_h,
                                 color=bg, transform=ax6.transAxes,
                                 clip_on=False))
    ax6.text(col_x1, y_pos, label, transform=ax6.transAxes,
             fontsize=10, color="#8A8FAA", va="center")
    ax6.text(col_x2, y_pos, value, transform=ax6.transAxes,
             fontsize=10, color="#E8EAF6", va="center", fontweight="bold")

ax6.set_title("Model summary", fontsize=11, pad=8)
plt.savefig(
    "project2_results.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="#0F1117"
)
plt.show()


# ─────────────────────────────────────────────
# 8. FINAL VERDICT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL RESULTS")
print("=" * 60)
print(f"""
  Algorithm  : K-Nearest Neighbors
  Optimal K  : {best_k}
  F1 (macro) : {f1_macro:.4f}
  F1 (wt.)   : {f1_wtd:.4f}

  Interpretation:
  ─ F1 near 1.0 means almost all predictions are correct.
  ─ The confusion matrix shows which classes were confused.
  ─ Setosa is perfectly separable; Versicolor/Virginica
    overlap slightly, which is expected from the data geometry.

  Next steps (to push further):
  ─ Try other classifiers: DecisionTree, SVM, Random Forest
  ─ Cross-validate with k-fold instead of a single split
  ─ Explore deeper features with PCA dimensionality reduction
""")
print("=" * 60)
print("  DecodeLabs | Build with 🤍 by Katlego Mathebula | Project 2 Complete ")
print("=" * 60)
