# Unsupervised Learning — Revision Notes
### AI-2002 | FAST NUCES

---

## Table of Contents
1. [Supervised vs Unsupervised](#1-supervised-vs-unsupervised)
2. [Types of Unsupervised Learning](#2-types-of-unsupervised-learning)
3. [K-Means Clustering](#3-k-means-clustering)
4. [Elbow Method](#4-elbow-method)
5. [Scaling in Clustering](#5-scaling-in-clustering)
6. [Full Code Template](#6-full-code-template)
7. [Quick Reference Cheatsheet](#7-quick-reference-cheatsheet)

---

## 1. Supervised vs Unsupervised

| | Supervised | Unsupervised |
|---|---|---|
| Labels | Has X and y | Only X — no labels |
| Goal | Predict correct output | Find hidden patterns |
| Train/Test Split | Required | Not needed |
| Pattern | fit → predict | fit_predict |
| Evaluation | Accuracy, R², F1 | WCSS, Elbow curve |
| Examples | Loan approval, score prediction | Customer segmentation, grouping |

### Why no train/test split?
All data is unlabeled — there's nothing to "test" against. No correct answers exist. The algorithm uses all available data to find groups.

### The new pattern — fit_predict
```python
# Supervised (two steps, two datasets)
model.fit(X_train, y_train)
model.predict(X_test)

# Unsupervised (one step, one dataset)
y_pred = model.fit_predict(X)  # learns clusters + assigns labels in one shot
```

---

## 2. Types of Unsupervised Learning

| Type | What it does | Example |
|---|---|---|
| **Clustering** | Groups similar data points together | Customer segmentation |
| **Association** | Finds relationships between variables | Market basket analysis (bread → butter) |
| **Dimensionality Reduction** | Reduces features while keeping information | PCA |

> Lab 11 focuses entirely on **Clustering → K-Means**

---

## 3. K-Means Clustering

**Use when:** you want to group similar data points without any labels

### The Idea
You tell the algorithm how many groups (K) you want. It figures out the groups itself by finding natural centers in the data.

```
K=3 → find 3 natural groups

Before:          After:
● ● ■ ■ ▲       🔴🔴 🔵🔵 🟢
  ● ■ ▲ ▲         🔴 🔵 🟢🟢
```

### How it Works (Step by Step)
```
Step 1: Pick K random points as starting centroids
Step 2: Assign every data point to its nearest centroid
Step 3: Recalculate centroid = mean of all points in that cluster
Step 4: Reassign points to new nearest centroids
Step 5: Repeat until nothing changes → clusters are stable
```

> Think of centroids as magnets that keep moving until they settle in the most natural center of each group.

### WCSS — Within Cluster Sum of Squares
Measures how tight your clusters are:

$$WCSS = \sum_{i} (distance(point_i, \ centroid))^2$$

| WCSS | Meaning |
|---|---|
| Low | Points are close to centroid → tight clusters ✅ |
| High | Points are spread from centroid → loose clusters ❌ |

`kmeans.inertia_` stores the WCSS value after fitting.

### Key Parameters

| Parameter | Meaning |
|---|---|
| `n_clusters` | How many clusters K |
| `init='k-means++'` | Smarter starting centroids → faster, better convergence |
| `random_state=42` | Reproducibility |
| `inertia_` | WCSS value after fitting |
| `fit_predict(X)` | Fit + assign cluster labels in one shot |

---

## 4. Elbow Method

You don't always know what K to use. The elbow method finds the optimal K by:
- Running K-Means for K=1 to K=10
- Calculating WCSS for each K
- Plotting WCSS vs K
- Picking K at the **elbow point** — where WCSS stops dropping sharply

```
WCSS
|*
| *
|  *
|    *               ← elbow here = optimal K
|        * * * * *
└──────────────────── K
  1  2  3  4  5  6
```

> After the elbow, adding more clusters barely reduces WCSS — not worth the complexity.
> Keep range(1, 11) — extending to larger ranges doesn't change where the elbow is, just adds a longer flat tail.

### Code
```python
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)   # store WCSS for this K

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.show()
# Read the plot → pick K at the elbow
```

---

## 5. Scaling in Clustering

Scaling matters **even more** in clustering than in supervised learning.

**Why?** Clustering uses only **distances** to group points. No labels exist to guide it. If features have very different ranges:

```
Income:        20,000 – 500,000   ← huge range
Spending Score:     1 – 100       ← tiny range
```

Income will completely dominate the distance calculation — the model basically ignores spending score. Clusters will be shaped entirely by income.

```python
# Without scaling → income dominates → misleading clusters
# With scaling    → all features contribute equally → meaningful clusters
```

### Selective Scaling
Sometimes tasks ask you to scale all features **except** one (e.g. age, or a categorical column):

```python
# Scale all except 'Age'
cols_to_scale = [col for col in X.columns if col != 'Age']
X_partial = X.copy()
X_partial[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
```

---

## 6. Full Code Template

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── LOAD ──────────────────────────────────────────────────────
df = pd.read_csv('data.csv')
print(df.info())
print(df.head())

# ── PREPROCESS ────────────────────────────────────────────────

# Drop ID — not a useful feature
df = df.drop(columns=['customer_id'])

# Strip whitespace (good habit)
df.columns = df.columns.str.strip()
df = df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)

# Encode categoricals
df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

# Select features for clustering
X = df.copy()   # or df[['col1', 'col2', 'col3']] for specific columns

# ── SCALE ─────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── ELBOW METHOD ──────────────────────────────────────────────
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.show()
# Read plot → pick K at elbow

# ── K-MEANS MODEL ─────────────────────────────────────────────
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_pred = kmeans.fit_predict(X_scaled)   # assigns cluster label 0,1,2... to each row

# Add labels back to original df
df['Cluster'] = y_pred
print(df.head())

# ── VISUALIZE ─────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x='Annual Income (k$)',       # x axis feature
    y='Spending Score (1-100)',   # y axis feature
    hue='Cluster',                # color by cluster
    palette='Set1',
    s=100
)
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# ── WITHOUT SCALING (for comparison) ─────────────────────────
kmeans_unscaled = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_pred_unscaled = kmeans_unscaled.fit_predict(X)   # raw X, no scaling

df['Cluster_Unscaled'] = y_pred_unscaled

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster_Unscaled',
    palette='Set1',
    s=100
)
plt.title('Customer Segments — Without Scaling')
plt.show()
```

---

## 7. Quick Reference Cheatsheet

### Preprocessing Rules for Clustering

| Rule | Why |
|---|---|
| Drop ID columns | Just a row number, meaningless for grouping |
| Encode categoricals | K-Means only works on numbers |
| Scale features | Distances dominate — scaling ensures equal contribution |
| No train/test split | No labels, no held-out test data needed |

### The Pipeline

```
Load → Drop ID → Strip → Encode → Select Features
→ Scale → Elbow Method → Pick K → fit_predict → Visualize
```

### Scaling Options

```python
# Scale everything
X_scaled = scaler.fit_transform(X)

# Scale everything except one column
cols_to_scale = [col for col in X.columns if col != 'Age']
X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

# No scaling (for comparison)
y_pred = kmeans.fit_predict(X)   # raw X
```

### Visualization Options

```python
# Simple scatter — 2 features
sns.scatterplot(data=df, x='feature1', y='feature2', hue='Cluster', palette='Set1')

# With centroids marked
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300, c='yellow', marker='*', label='Centroid'
)
```

### Common Mistakes

| Mistake | Fix |
|---|---|
| Not dropping ID column | Always drop before selecting X |
| Not scaling before clustering | K-Means is distance-based — scale everything |
| Picking K too large | Use elbow method, keep range(1,11) |
| Not encoding categoricals | K-Means needs numbers only |
| fit_transform on clustering | Just `scaler.fit_transform(X)` — no train/test split |
| Forgetting `df['Cluster'] = y_pred` | Always add labels back to df for visualization |
