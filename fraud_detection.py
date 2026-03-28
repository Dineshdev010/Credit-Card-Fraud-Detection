"""
PROJECT 5: Credit Card Fraud Detection
=========================================
Tools  : Python, Pandas, Scikit-learn, Seaborn, imbalanced-learn (optional)
Dataset: Synthetic anonymized transaction data (generated here)
         (For real data: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
Run    : python fraud_detection.py

What this project covers:
  - Understanding highly imbalanced datasets
  - EDA on fraud patterns
  - Feature scaling
  - SMOTE for oversampling (if imbalanced-learn installed)
  - Logistic Regression, Random Forest, Decision Tree
  - Precision, Recall, F1, ROC-AUC evaluation
  - Confusion matrix
  - ROC curve comparison
  - Feature importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, f1_score
)
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="deep")
DARK = "#2c3e50"
FRAUD_COLOR   = "#e74c3c"
NORMAL_COLOR  = "#3498db"

# Optional SMOTE
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

print("=" * 55)
print("  PROJECT 5: Credit Card Fraud Detection")
print("=" * 55)

print("\n[1/8] Generating transaction dataset...")

np.random.seed(42)
N        = 50000
N_FRAUD  = 500    # ~1% fraud rate — realistic imbalance

# Normal transactions
n_normal = N - N_FRAUD
amt_normal = np.random.exponential(scale=80, size=n_normal).clip(1, 5000)
hour_normal = np.random.randint(0, 24, n_normal)
v_features_normal = np.random.randn(n_normal, 28)

# Fraud transactions — different distribution
amt_fraud = np.random.exponential(scale=250, size=N_FRAUD).clip(1, 4000)
hour_fraud = np.random.choice(range(0, 6), N_FRAUD)   # often late night
v_features_fraud = np.random.randn(N_FRAUD, 28) * 1.8  # more extreme values

# Combine
amounts = np.concatenate([amt_normal, amt_fraud])
hours   = np.concatenate([hour_normal, hour_fraud])
v_data  = np.vstack([v_features_normal, v_features_fraud])
labels  = np.array([0] * n_normal + [1] * N_FRAUD)

# Shuffle
idx = np.random.permutation(N)
amounts = amounts[idx]
hours   = hours[idx]
v_data  = v_data[idx]
labels  = labels[idx]

# Build DataFrame
v_cols = [f"V{i}" for i in range(1, 29)]
df = pd.DataFrame(v_data, columns=v_cols)
df["Amount"] = amounts.round(2)
df["Hour"]   = hours
df["Class"]  = labels   # 0 = Normal, 1 = Fraud

fraud_count  = (df["Class"] == 1).sum()
normal_count = (df["Class"] == 0).sum()
fraud_pct    = fraud_count / N * 100

print(f"   Total transactions : {N:,}")
print(f"   Fraudulent         : {fraud_count:,} ({fraud_pct:.2f}%)")
print(f"   Normal             : {normal_count:,} ({100-fraud_pct:.2f}%)")

print("\n[2/8] Exploratory Data Analysis...")

fraud   = df[df["Class"] == 1]
normal  = df[df["Class"] == 0]

print(f"\n   Fraud transaction stats:")
print(f"   Avg amount (fraud)  : ${fraud['Amount'].mean():.2f}")
print(f"   Avg amount (normal) : ${normal['Amount'].mean():.2f}")
print(f"   Fraud peak hour     : {fraud['Hour'].value_counts().index[0]}:00")


print("\n[3/8] Preparing features...")

features = v_cols + ["Amount", "Hour"]
X = df[features].copy()
y = df["Class"]

scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])
X["Hour"]   = scaler.fit_transform(X[["Hour"]])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"   Train size : {len(X_train):,}")
print(f"   Test size  : {len(X_test):,}")
print(f"   Fraud in test : {y_test.sum()} ({y_test.sum()/len(y_test)*100:.2f}%)")

# SMOTE
if SMOTE_AVAILABLE:
    print("\n   Applying SMOTE oversampling...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"   After SMOTE — Fraud: {y_train_res.sum():,}  Normal: {(y_train_res==0).sum():,}")
else:
    X_train_res, y_train_res = X_train, y_train
    print("\n   SMOTE not available (pip install imbalanced-learn)")
    print("   Using class_weight='balanced' instead")


print("\n[4/8] Training models...")

models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=500, random_state=42),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced", max_depth=8, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, class_weight="balanced",
        random_state=42, n_jobs=-1),
}

results = {}
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    ap   = average_precision_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

    results[name] = {
        "model": model, "y_pred": y_pred, "y_prob": y_prob,
        "f1": f1, "auc": auc, "ap": ap, "cm": cm,
        "precision": precision, "recall": recall,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }
    print(f"\n   {name}:")
    print(f"     Precision : {precision:.4f}")
    print(f"     Recall    : {recall:.4f}")
    print(f"     F1 Score  : {f1:.4f}")
    print(f"     ROC-AUC   : {auc:.4f}")
    print(f"     Avg Prec  : {ap:.4f}")
    print(f"     TP={tp}  FP={fp}  FN={fn}  TN={tn}")

best_model_name = max(results, key=lambda k: results[k]["f1"])
print(f"\n   Best model (F1): {best_model_name}")


print(f"\n[5/8] Classification report — {best_model_name}:")
print(classification_report(y_test, results[best_model_name]["y_pred"],
                            target_names=["Normal","Fraud"]))

print("\n[6/8] Creating visualizations...")

fig, axes = plt.subplots(3, 3, figsize=(19, 15))
fig.suptitle("Credit Card Fraud Detection — Model Analysis",
             fontsize=18, fontweight="bold", color=DARK, y=0.99)
plt.subplots_adjust(hspace=0.45, wspace=0.35)

# 1. Class imbalance
axes[0,0].bar(["Normal","Fraud"], [normal_count, fraud_count],
              color=[NORMAL_COLOR, FRAUD_COLOR], edgecolor="white")
axes[0,0].set_ylabel("Count")
axes[0,0].set_title("Class Imbalance", fontweight="bold")
axes[0,0].set_yscale("log")
for i, v in enumerate([normal_count, fraud_count]):
    axes[0,0].text(i, v * 1.1, f"{v:,}", ha="center", fontsize=10)

# 2. Amount distribution
axes[0,1].hist(normal["Amount"], bins=50, alpha=0.6,
               color=NORMAL_COLOR, label="Normal", density=True)
axes[0,1].hist(fraud["Amount"], bins=50, alpha=0.6,
               color=FRAUD_COLOR, label="Fraud", density=True)
axes[0,1].set_xlabel("Transaction Amount ($)")
axes[0,1].set_ylabel("Density")
axes[0,1].set_title("Amount Distribution", fontweight="bold")
axes[0,1].legend()

# 3. Fraud by hour
hour_fraud_counts = fraud["Hour"].value_counts().sort_index()
hour_all_counts   = df.groupby("Hour")["Class"].mean() * 100
axes[0,2].bar(hour_all_counts.index, hour_all_counts.values,
              color=FRAUD_COLOR, alpha=0.8)
axes[0,2].set_xlabel("Hour of Day")
axes[0,2].set_ylabel("Fraud Rate (%)")
axes[0,2].set_title("Fraud Rate by Hour", fontweight="bold")

# 4. Confusion matrices (best model)
r = results[best_model_name]
cm_pct = r["cm"].astype(float) / r["cm"].sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Reds",
            ax=axes[1,0], linewidths=0.5,
            xticklabels=["Normal","Fraud"],
            yticklabels=["Normal","Fraud"],
            cbar_kws={"label": "%"})
axes[1,0].set_xlabel("Predicted")
axes[1,0].set_ylabel("Actual")
axes[1,0].set_title(f"Confusion Matrix\n({best_model_name})", fontweight="bold")

# 5. ROC curves
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
    axes[1,1].plot(fpr, tpr, linewidth=2,
                   label=f"{name} (AUC={r['auc']:.3f})")
axes[1,1].plot([0,1],[0,1],"k--", linewidth=1)
axes[1,1].set_xlabel("False Positive Rate")
axes[1,1].set_ylabel("True Positive Rate")
axes[1,1].set_title("ROC Curve Comparison", fontweight="bold")
axes[1,1].legend(fontsize=8)

# 6. Precision-Recall curves
for name, r in results.items():
    prec, rec, _ = precision_recall_curve(y_test, r["y_prob"])
    axes[1,2].plot(rec, prec, linewidth=2,
                   label=f"{name} (AP={r['ap']:.3f})")
axes[1,2].set_xlabel("Recall")
axes[1,2].set_ylabel("Precision")
axes[1,2].set_title("Precision-Recall Curve", fontweight="bold")
axes[1,2].legend(fontsize=8)

# 7. Model comparison bar chart
model_names = list(results.keys())
metrics     = ["f1","auc","precision","recall"]
metric_vals = {m: [results[n][m] for n in model_names] for m in metrics}

x    = np.arange(len(model_names))
w    = 0.20
colors_m = ["#3498db","#2ecc71","#f39c12","#e74c3c"]
for i, (m, vals) in enumerate(metric_vals.items()):
    axes[2,0].bar(x + i*w, vals, w, label=m.upper(), color=colors_m[i],
                  alpha=0.85, edgecolor="white")
axes[2,0].set_xticks(x + w*1.5)
axes[2,0].set_xticklabels([n.replace(" ","\n") for n in model_names], fontsize=8)
axes[2,0].set_ylabel("Score")
axes[2,0].set_ylim(0, 1.1)
axes[2,0].set_title("Model Metric Comparison", fontweight="bold")
axes[2,0].legend(fontsize=7)

# 8. Random Forest feature importance
rf = results["Random Forest"]["model"]
feat_imp = pd.Series(rf.feature_importances_, index=features)
top_feats = feat_imp.sort_values(ascending=False).head(12)
axes[2,1].barh(top_feats.index[::-1], top_feats.values[::-1],
               color="#9b59b6", alpha=0.85)
axes[2,1].set_xlabel("Importance")
axes[2,1].set_title("Top 12 Feature Importances\n(Random Forest)", fontweight="bold")

# 9. Fraud probability distribution
r = results["Random Forest"]
axes[2,2].hist(r["y_prob"][y_test == 0], bins=40, alpha=0.6,
               color=NORMAL_COLOR, label="Normal", density=True)
axes[2,2].hist(r["y_prob"][y_test == 1], bins=40, alpha=0.6,
               color=FRAUD_COLOR, label="Fraud", density=True)
axes[2,2].set_xlabel("Predicted Fraud Probability")
axes[2,2].set_ylabel("Density")
axes[2,2].set_title("Fraud Score Distribution\n(Random Forest)", fontweight="bold")
axes[2,2].legend()

plt.savefig("fraud_detection_dashboard.png", dpi=130, bbox_inches="tight")
print("   Saved → fraud_detection_dashboard.png")

print("\n[7/8] Cost-benefit analysis...")
avg_fraud_amount = fraud["Amount"].mean()
r = results["Random Forest"]
cost_fn = r["fn"] * avg_fraud_amount   # missed frauds = lost money
cost_fp = r["fp"] * 2                  # false alerts = ~$2 investigation cost
benefit = r["tp"] * avg_fraud_amount   # caught frauds = money saved

print(f"   Avg fraud amount    : ${avg_fraud_amount:.2f}")
print(f"   Fraud caught (TP)   : {r['tp']} → saved ${benefit:,.0f}")
print(f"   Fraud missed (FN)   : {r['fn']} → lost  ${cost_fn:,.0f}")
print(f"   False alerts (FP)   : {r['fp']} → cost  ${cost_fp:,.0f}")
print(f"   Net benefit         : ${benefit - cost_fn - cost_fp:,.0f}")

print("\n[8/8] Key Insights:")
print(f"   1. Dataset is highly imbalanced: {fraud_pct:.2f}% fraud")
print(f"   2. Fraud transactions have higher average amounts (${fraud['Amount'].mean():.0f} vs ${normal['Amount'].mean():.0f})")
print(f"   3. Fraud peaks at night hours (0–5 AM)")
print(f"   4. Best model: {best_model_name} (F1={results[best_model_name]['f1']:.4f})")
print(f"   5. Use Recall over Accuracy — missing fraud is costlier than false alerts")
print(f"   6. SMOTE/class_weight helps the model learn the minority class")
print(f"   7. Precision-Recall curve is more informative than ROC for imbalanced data")

df.to_csv("transactions_clean.csv", index=False)
print("\n   Saved → transactions_clean.csv")
print("\n✅ Project 5 complete!")
plt.show()
