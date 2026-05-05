# Supervised Learning — Revision Notes
### AI-2002 | FAST NUCES

---

## Table of Contents
1. [Machine Learning Overview](#1-machine-learning-overview)
2. [Preprocessing](#2-preprocessing)
3. [Linear Regression](#3-linear-regression)
4. [Logistic Regression](#4-logistic-regression)
5. [Decision Tree](#5-decision-tree)
6. [Support Vector Machine (SVM)](#6-support-vector-machine-svm)
7. [Random Forest](#7-random-forest)
8. [Quick Reference Cheatsheet](#8-quick-reference-cheatsheet)

---

## 1. Machine Learning Overview

**Machine Learning** — systems that learn from data without being explicitly programmed.

| Type | What it does | Examples |
|---|---|---|
| **Supervised** | Learns from labelled data (input + correct output) | Classification, Regression |
| **Unsupervised** | Finds patterns in unlabelled data | K-Means Clustering |
| **Reinforcement** | Learns by reward/penalty | Game AI, Recommender systems |

### Supervised Learning — Two Types

| Type | When to use | Target variable | Examples |
|---|---|---|---|
| **Classification** | Output is a category | Discrete (0/1, Yes/No) | Spam detection, Loan approval |
| **Regression** | Output is a number | Continuous | Score prediction, House price |

### Python Libraries Used

| Library | Purpose |
|---|---|
| `numpy` | Numerical computations, arrays |
| `pandas` | Data manipulation, DataFrames |
| `matplotlib` | Data visualization |
| `seaborn` | Advanced plotting |
| `scikit-learn` | ML algorithms, preprocessing, metrics |

---

## 2. Preprocessing

> **Golden Rule: Encode → Split → Scale**
> Never scale before splitting — that causes data leakage.

### Why this order?
- **Encode first** — ML models only understand numbers, not strings
- **Split before Scale** — test data must remain "unseen", even by the scaler
- **Scale after Split** — fit scaler only on training data, apply to both

---

### 2.1 Encoding Categorical Variables

**Two types of categorical data:**

| Type | Definition | Example | Encoder |
|---|---|---|---|
| **Ordinal** | Has a natural order | Low < Medium < High | `LabelEncoder` |
| **Nominal** | No natural order | Male/Female, City names | `pd.get_dummies()` |

#### LabelEncoder — for ordinal data
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Loan_Term'] = le.fit_transform(df['Loan_Term'])
# Month-to-Month=0, One Year=1, Two Year=2
```
> ⚠️ LabelEncoder assigns numbers alphabetically by default. Make sure alphabetical order matches actual order, or handle manually.

#### get_dummies — for nominal data
```python
df = pd.get_dummies(df, columns=['Employment_Status'], drop_first=True)
```
> `drop_first=True` — drops one column to avoid the **Dummy Variable Trap**.
> When all dummy columns are 0, the model infers the dropped category.
>
> Example: `Self-Employed=0, Unemployed=0` → means **Employed** (the dropped one)

---

### 2.2 Train-Test Split

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=['target_column'])
y = df['target_column']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 20% test, 80% train
    random_state=42   # fixed seed → same split every run
)
```

---

### 2.3 Feature Scaling

**Why scale?** — Without scaling, features with larger ranges (e.g. income: 50,000) dominate features with smaller ranges (e.g. num_children: 2), even if num_children is more important.

| Scaler | Formula | Use when |
|---|---|---|
| `StandardScaler` | z = (x − mean) / std → mean=0, std=1 | General purpose, SVM, Logistic Regression |
| `MinMaxScaler` | x = (x − min) / (max − min) → range [0,1] | Neural networks, when you need strict 0–1 range |

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # learns mean/std + applies
X_test  = scaler.transform(X_test)       # applies only (no learning)
```

#### fit_transform vs transform

| Method | What it does | Use on |
|---|---|---|
| `fit_transform` | Learns statistics + applies transformation | Training data only |
| `transform` | Applies transformation using already-learned statistics | Test data + new data |

> **Data Leakage** — if you `fit_transform` on test data, the scaler learns from it. Test data is supposed to simulate unseen real-world data. If your pipeline learns from it, your evaluation is lying to you.

---

### 2.4 Predicting on New Data

New data must go through **the same transformations** as training data.

```python
# Must match exact column structure after encoding
new_applicant = pd.DataFrame([{
    'Income': 50000,
    'Credit_Score': 710,
    'Loan_Amount': 12000,
    'Existing_Debt': 2000,
    'Loan_Term': 1,                          # already encoded
    'Employment_Status_Self-Employed': 0,
    'Employment_Status_Unemployed': 0,       # 0,0 = Employed
    'Marital_Status_Married': 1,
    'Marital_Status_Single': 0
}])

# transform only — never fit_transform on new data
new_scaled = scaler.transform(new_applicant)
prediction  = model.predict(new_scaled)
```

---

## 3. Linear Regression

**Use when:** target variable is **continuous** (predicting a number)

**Examples:** exam score, house price, salary, temperature

### The Idea
Find the best-fit line through data points:

```
y = B₀ + B₁X

B₀ = intercept (where line crosses y-axis)
B₁ = slope (how much y changes per unit of X)
Goal = minimize error between predicted and actual values
```

### Code
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.2f}")
```

### Evaluation Metrics

| Metric | Formula | Meaning |
|---|---|---|
| **MAE** | avg(|actual − predicted|) | Average error in same unit as target |
| **RMSE** | √(avg((actual − predicted)²)) | Punishes big errors more (sensitive to outliers) |
| **R²** | 1 − (residuals² / total variance) | % of variance explained. Closer to 1 = better |

> **MAE vs RMSE gap** — if RMSE is much larger than MAE, a few predictions were very wrong (outliers dragging RMSE up). MAE treats all errors equally so it stays lower.
>
> **R² = 0.2** → bad. Model explains only 20% of variance.
> **R² = 0.85** → good. Model explains 85% of variance.

---

## 4. Logistic Regression

**Use when:** target variable is **categorical** (binary 0/1 classification)

**Examples:** loan approved/rejected, spam/not spam, churn/no churn

> Despite the name, this is a **classification** algorithm, not regression.

### The Idea
Uses the **sigmoid function** to output a probability between 0 and 1:

```
P(y=1) = 1 / (1 + e^(-z))

Output > 0.5 → Class 1 (Approved)
Output < 0.5 → Class 0 (Rejected)
```

### Code
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)          # hard class: 0 or 1
y_prob = log_model.predict_proba(X_test)    # probabilities: [[0.3, 0.7], ...]

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall:    {recall_score(y_test, y_pred):.2f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.2f}")

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()
```

### Evaluation Metrics

#### Confusion Matrix
```
                Predicted 0     Predicted 1
Actual 0       True Neg (TN)   False Pos (FP)
Actual 1       False Neg (FN)  True Pos (TP)
```

| Metric | Formula | Meaning | Care about when |
|---|---|---|---|
| **Accuracy** | (TP+TN) / all | Overall correctness | Balanced datasets |
| **Precision** | TP / (TP+FP) | Of all predicted +ve, how many were actually +ve | False positives are costly (spam filter) |
| **Recall** | TP / (TP+FN) | Of all actual +ve, how many did you catch | False negatives are costly (cancer/fraud) |
| **F1 Score** | 2 × (P×R) / (P+R) | Balance of precision & recall | Need single metric for both |

> **Trap:** High Accuracy + Low Recall = lazy model predicting majority class.
> Example: 98% accuracy on fraud detection with Recall=0.10 means it's catching almost zero actual frauds — it's just predicting "not fraud" for everything.

### predict() vs predict_proba()
```python
model.predict(X_test)           # → [0, 1, 1, 0]          hard class label
model.predict_proba(X_test)     # → [[0.77, 0.23], ...]    probability of each class
                                #   [P(class=0), P(class=1)]
```

---

## 5. Decision Tree

**Use when:** classification or regression, need interpretability

### The Idea
Asks a series of yes/no questions to arrive at a prediction:
```
Is Credit_Score > 700?
├── Yes → Is Income > 50000?
│         ├── Yes → Approved ✅
│         └── No  → Rejected ❌
└── No  → Rejected ❌
```

The tree learns which questions to ask and in what order from training data (based on which split reduces uncertainty the most — highest **information gain**).

### Key Terminology

| Term | Meaning |
|---|---|
| **Root Node** | First question — most important feature (highest information gain) |
| **Decision Node** | A question in the middle of the tree |
| **Leaf Node** | Final answer — no more questions |
| **Splitting** | Dividing data at each node |
| **Branch** | A path from one node to another |
| **Depth** | Number of levels in the tree |

### Code
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))  # precision, recall, F1 in one shot
```

### Feature Importance
```python
importance = pd.Series(
    dt_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(importance)
# Credit_Score  0.85  ← responsible for 85% of predictions
# Income        0.10
# ...

importance.plot(kind='bar')
plt.title('Feature Importance')
plt.show()
```

### Visualizing the Tree
```python
plt.figure(figsize=(12, 6))
plot_tree(dt_model,
          feature_names=X.columns,
          class_names=['Rejected', 'Approved'],
          filled=True,     # color coded by class
          max_depth=3)     # limit depth for readability
plt.show()
```

### Overfitting Problem
```python
# Default — grows until it memorizes training data
dt = DecisionTreeClassifier()
# Training Accuracy: 100% ← memorized everything ❌
# Testing Accuracy:  60%  ← can't generalize ❌

# Fix — limit depth
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
# Training Accuracy: 88% ✅
# Testing Accuracy:  85% ✅
```

---

## 6. Support Vector Machine (SVM)

**Use when:** classification, works well on smaller/medium datasets

### The Idea
Finds the **hyperplane** (decision boundary) that separates classes with the **maximum margin**.

```
   Class 0 ●  ●              ■  ■ Class 1
            ●    ●          ■    ■
                   |margin|
                  hyperplane
                (decision boundary)
```

- **Hyperplane** — the boundary separating classes
- **Margin** — gap between the hyperplane and the nearest data points of each class
- **Support Vectors** — the closest data points to the hyperplane. They literally define it. Remove any other point and boundary stays the same. Remove a support vector and it shifts.
- **Goal** — maximize the margin

### Linear vs Non-Linear
```python
svm = SVC(kernel='linear')  # data cleanly separable by straight line
svm = SVC(kernel='rbf')     # data not separable → maps to higher dimension
```
> Think of it like this: if points are mixed in 2D, lift them into 3D where they can be separated by a flat plane. That's what the kernel does.

### Key Parameters

| Parameter | Meaning | Effect |
|---|---|---|
| `kernel` | Type of boundary | `linear`, `rbf` (most common), `poly` |
| `C` | Strictness of boundary | High C = less margin, risks overfitting |
| `gamma` | Influence range of each point | High gamma = only nearby points matter |
| `probability=True` | Enable predict_proba | Required for probability output |

### Code
```python
from sklearn.svm import SVC

svm_model = SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

print(f"Loan Approved:        {svm_model.predict(new_scaled)[0]}")
print(f"Approval Probability: {svm_model.predict_proba(new_scaled)[0][1]:.2f}")
```

### Decision Boundary Visualization
```python
from sklearn.inspection import DecisionBoundaryDisplay

# Can't visualize N dimensions → use 2 features only
X_vis   = X_train[:, :2]
svm_vis = SVC(kernel='rbf', C=1, gamma='scale')
svm_vis.fit(X_vis, y_train)

DecisionBoundaryDisplay.from_estimator(
    svm_vis, X_vis,
    response_method='predict',
    cmap='coolwarm', alpha=0.5
)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_train, cmap='coolwarm', edgecolors='k')
plt.title('SVM — Decision Boundary')
plt.show()
```

---

## 7. Random Forest

**Use when:** classification, want better accuracy + less overfitting than a single Decision Tree

### The Idea
Instead of one Decision Tree, build **100 trees** each trained on a **random subset** of data and features. All trees vote — majority wins.

```
Tree 1  → Approved ✅
Tree 2  → Rejected ❌
Tree 3  → Approved ✅
Tree 4  → Approved ✅
────────────────────
Final   → Approved ✅  (majority vote)
```

Because each tree sees different data and features, no single tree dominates. The vote averages out individual mistakes → **less overfitting**.

### Decision Tree vs Random Forest

| | Decision Tree | Random Forest |
|---|---|---|
| Trees | 1 | 100+ |
| Overfitting | High risk | Low risk |
| Interpretability | Easy (visualize) | Hard |
| Performance | Decent | Usually better |
| Speed | Fast | Slower |

### Code
```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

# Feature importance — same pattern as Decision Tree
importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

importance.plot(kind='bar')
plt.title('Random Forest — Feature Importance')
plt.show()
```

---

## 8. Quick Reference Cheatsheet

### When to Use What

| Model | Target | Best for |
|---|---|---|
| Linear Regression | Continuous | Predicting numbers (score, price) |
| Logistic Regression | Binary 0/1 | Simple classification, need probabilities |
| Decision Tree | Both | Need interpretability, visualize rules |
| SVM | Categorical | Smaller datasets, complex boundaries |
| Random Forest | Categorical | Best accuracy, don't need interpretability |

### Model Import + Initialize

```python
from sklearn.linear_model  import LinearRegression
from sklearn.linear_model  import LogisticRegression
from sklearn.tree          import DecisionTreeClassifier
from sklearn.svm           import SVC
from sklearn.ensemble      import RandomForestClassifier

lr  = LinearRegression()
log = LogisticRegression()
dt  = DecisionTreeClassifier(random_state=42)
svm = SVC(kernel='rbf', probability=True, random_state=42)
rf  = RandomForestClassifier(n_estimators=100, random_state=42)
```

### Universal Model Pattern

```python
# 1. Train
model.fit(X_train, y_train)

# 2. Predict
y_pred = model.predict(X_test)

# 3. Evaluate
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 4. New data
new_scaled = scaler.transform(new_data)
print(model.predict(new_scaled))
print(model.predict_proba(new_scaled)[0][1])  # probability of class 1
```

### Metrics at a Glance

| Metric | Function | Use for |
|---|---|---|
| MAE | `mean_absolute_error(y_test, y_pred)` | Regression |
| RMSE | `np.sqrt(mean_squared_error(y_test, y_pred))` | Regression |
| R² | `r2_score(y_test, y_pred)` | Regression |
| Accuracy | `accuracy_score(y_test, y_pred)` | Classification |
| Precision | `precision_score(y_test, y_pred)` | Classification |
| Recall | `recall_score(y_test, y_pred)` | Classification |
| F1 | `f1_score(y_test, y_pred)` | Classification |
| All-in-one | `classification_report(y_test, y_pred)` | Classification |
| Confusion Matrix | `confusion_matrix(y_test, y_pred)` | Classification |

### Key Parameters to Know

| Parameter | Model | Meaning |
|---|---|---|
| `random_state=42` | All | Fixes randomness for reproducibility |
| `test_size=0.2` | train_test_split | 20% for testing |
| `drop_first=True` | get_dummies | Avoid dummy variable trap |
| `max_depth=4` | Decision Tree | Prevents overfitting |
| `kernel='rbf'` | SVM | Non-linear boundary |
| `probability=True` | SVM | Enables predict_proba |
| `n_estimators=100` | Random Forest | Number of trees |

### Common Mistakes

| Mistake | Fix |
|---|---|
| Scale before split | Always split first, then scale |
| `fit_transform` on test data | Use `transform` only on test/new data |
| Encoding same column twice | Pick LabelEncoder OR get_dummies, not both |
| `df[:-1]` to drop last column | Use `df.drop(columns=['col'])` |
| Leaving ID columns in X | Drop ID/index columns before training |
| Raw strings in new data | Encode new data the same way as training data |
| No `probability=True` in SVM | Add it if you need `predict_proba` |
| Decision Tree without max_depth | Set `max_depth` to prevent overfitting |
