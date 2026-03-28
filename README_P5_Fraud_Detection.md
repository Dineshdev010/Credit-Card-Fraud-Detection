# Project 5 — Credit Card Fraud Detection 💳

## What Is This Project?

The most advanced project in the series — and one of the most common interview
challenges for data analyst and data science roles. It tackles the hardest
problem in machine learning: classification when 99% of your data is one class.

Standard accuracy is completely useless here. A model that always predicts
"not fraud" achieves 99% accuracy but catches zero frauds. This project
teaches exactly what metrics, techniques, and thinking are needed for
real-world imbalanced datasets.

**Difficulty:** Advanced
**Time:** 5–7 days
**Tools:** Python, Pandas, Scikit-learn, Seaborn, imbalanced-learn (optional)
**No external dataset needed** — but the real Kaggle dataset is a direct swap

---

## What Problem Does It Solve?

A bank processes 50,000 transactions per day. Roughly 500 of them are fraud.
The challenge is:
- Catching as many fraudulent transactions as possible (high Recall)
- Not flagging so many legitimate transactions that investigators are overwhelmed
  (reasonable Precision)
- Doing this even though the model has seen 99x more normal transactions
  than fraud during training

---

## What You Will Build

| File | Description |
|------|-------------|
| `fraud_detection_dashboard.png` | 9-panel model analysis dashboard |
| `transactions_clean.csv` | 50,000 transactions with labels |

### The 9 Dashboard Panels

| Panel | Chart Type | What It Shows |
|-------|-----------|---------------|
| 1 | Bar chart (log scale) | Class imbalance — how extreme is 1% fraud? |
| 2 | Histogram overlay | Transaction amount: fraud vs normal distribution |
| 3 | Bar chart | Fraud rate by hour of day — when does fraud peak? |
| 4 | Heatmap | Confusion matrix as percentages for the best model |
| 5 | Line chart | ROC curves for all 3 models on one plot |
| 6 | Line chart | Precision-Recall curves for all 3 models |
| 7 | Grouped bar | F1, AUC, Precision, Recall side-by-side per model |
| 8 | Horizontal bar | Top 12 feature importances from Random Forest |
| 9 | Histogram overlay | Fraud probability scores: fraud vs normal |

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
pip install imbalanced-learn   # optional — for SMOTE oversampling

python fraud_detection.py
```

If `imbalanced-learn` is not installed, the script automatically falls back
to `class_weight='balanced'` — no errors, just a different balancing method.

---

## Full Step-by-Step Explanation

### Step 1 — Generate the Dataset

Creates 50,000 transactions where fraudulent transactions have detectably
different patterns from normal ones:

```python
# Normal transactions
amt_normal  = np.random.exponential(scale=80, size=49500).clip(1, 5000)
hour_normal = np.random.randint(0, 24, 49500)  # spread throughout the day

# Fraud transactions — different distribution
amt_fraud  = np.random.exponential(scale=250, size=500).clip(1, 4000)  # higher amounts
hour_fraud = np.random.choice(range(0, 6), 500)  # concentrated at night (0–5 AM)
```

This mimics the V1–V28 PCA features in the real Kaggle dataset — the actual
transaction details are anonymised but retain their statistical relationships.

### Step 2 — Understand the Class Imbalance

Before any modelling, visualise just how imbalanced the data is:

```python
fraud_pct = fraud_count / N * 100
# Result: 1.00% — one fraudulent transaction per 99 normal ones
```

**Why this is a problem:** If a model predicts "not fraud" for every single
transaction, it achieves 99% accuracy. But it catches zero frauds and is
completely worthless. This is why accuracy is the wrong metric for this problem.

### Step 3 — Scale Features and Split Data

Always split before scaling:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y  # keeps class ratio the same in both splits
)

scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])
X["Hour"]   = scaler.fit_transform(X[["Hour"]])
```

`stratify=y` is critical for imbalanced data. Without it, you might
accidentally put all the fraud in training and none in test, or vice versa.

### Step 4 — Handle the Imbalance with SMOTE

SMOTE (Synthetic Minority Oversampling Technique) creates new synthetic
fraud examples by interpolating between existing ones:

```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
# Training set now has equal fraud and normal examples
```

**Critical rule:** SMOTE is applied ONLY to the training set. Never touch
the test set. Applying SMOTE to the test set would make your evaluation
results meaningless.

If SMOTE is unavailable, use `class_weight='balanced'` instead. This tells
sklearn to weight minority class errors more heavily during training —
simpler but effective.

### Step 5 — Train Three Models

```python
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced"),
    "Decision Tree":       DecisionTreeClassifier(class_weight="balanced", max_depth=8),
    "Random Forest":       RandomForestClassifier(n_estimators=100, class_weight="balanced"),
}
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
```

Each model has different strengths:
- **Logistic Regression:** Fast, interpretable, good baseline
- **Decision Tree:** Visual, easy to explain to non-technical stakeholders,
  but prone to overfitting
- **Random Forest:** Ensemble of decision trees — typically the best performer
  on tabular data

### Step 6 — Evaluate with the Right Metrics

```python
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # fraud probability score

# Confusion matrix breakdown
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
precision = tp / (tp + fp)  # of flagged frauds, how many are real
recall    = tp / (tp + fn)  # of real frauds, how many did we catch
```

### Step 7 — ROC Curve vs Precision-Recall Curve

Both curves are plotted for all 3 models:

```python
# ROC curve — good for comparing models
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc          = roc_auc_score(y_test, y_prob)

# Precision-Recall curve — better for imbalanced data
prec, rec, _ = precision_recall_curve(y_test, y_prob)
ap            = average_precision_score(y_test, y_prob)
```

The Precision-Recall curve is more informative than ROC for severely
imbalanced datasets because it focuses entirely on the minority class
(fraud) rather than averaging across both classes.

### Step 8 — Cost-Benefit Analysis

The final output translates model performance into business value:

```python
avg_fraud_amount = fraud["Amount"].mean()
benefit  = tp * avg_fraud_amount   # money saved by catching fraud
cost_fn  = fn * avg_fraud_amount   # money lost from missed fraud
cost_fp  = fp * 2                  # investigation cost per false alert (~$2)
net      = benefit - cost_fn - cost_fp
```

This is what matters to a bank — not the F1 score, but how much money the
model saves after accounting for investigation costs.

---

## Why Accuracy Is the Wrong Metric

| Metric | What It Measures | The Problem |
|--------|-----------------|-------------|
| Accuracy | % of all predictions correct | 99% accuracy = catch zero frauds |
| Precision | Of flagged frauds, how many are real | High = fewer false investigations |
| Recall | Of real frauds, how many were caught | High = fewer missed frauds, less financial loss |
| F1 Score | Harmonic mean of precision and recall | Best single metric for imbalanced problems |
| ROC-AUC | Class separability across all thresholds | Good comparison metric, optimistic with imbalance |
| PR-AUC | Area under precision-recall curve | Most honest metric for severely imbalanced data |

**The core tradeoff:** Increasing Recall (catching more fraud) always
decreases Precision (more false alerts). The right balance depends on
your cost model — how expensive is a missed fraud vs a false investigation?

---

## Key Concepts Explained

| Concept | Plain English |
|---------|--------------|
| SMOTE | Generates synthetic minority class samples by blending real examples |
| `class_weight='balanced'` | Sklearn weights minority class errors higher in the loss function |
| `stratify=y` in train_test_split | Keeps class ratio identical in both training and test splits |
| `predict_proba()[:,1]` | Returns the fraud probability score, not just a binary Yes/No |
| `confusion_matrix().ravel()` | Unpacks into (TN, FP, FN, TP) for direct arithmetic |
| `roc_curve()` | False Positive Rate vs True Positive Rate at every decision threshold |
| `precision_recall_curve()` | Precision vs Recall at every threshold — better for imbalance |
| `feature_importances_` | How much each feature reduced impurity across all trees |

---

## Key Insights You Will Discover

- Fraud transactions have roughly 3x higher average amounts than normal
- Fraud is heavily concentrated in late-night hours (midnight to 5 AM)
- Random Forest almost always outperforms Logistic Regression on recall
- The Precision-Recall curve tells a very different story than the ROC curve
- SMOTE significantly boosts recall — but at the cost of more false alerts
- The business value (money saved) is often the most convincing output for
  stakeholders who do not understand F1 scores

---

## Bonus Challenges

1. Use the real **Credit Card Fraud Detection** dataset from Kaggle — 284,807
   transactions with only 0.17% fraud (even more extreme imbalance)
2. Tune the decision threshold away from 0.5 to match your cost function
3. Add **XGBoost** or **LightGBM** — compare vs Random Forest
4. Try **Isolation Forest** for unsupervised anomaly detection without labels
5. Build a Flask API that scores a new transaction in real time

---

## Real Dataset

**Kaggle:** Search "Credit Card Fraud Detection" by mlg-ulb
284,807 transactions from European cardholders in September 2013.
Features V1–V28 are PCA-transformed for anonymity. The class imbalance is
0.17% — even more extreme than this project.
