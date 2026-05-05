# ══════════════════════════════════════════════════════════════
#  SUPERVISED LEARNING — REVISION
#  Topics: Preprocessing, Linear Regression, Logistic Regression,
#          Decision Tree, SVM, Random Forest
# ══════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing        import LabelEncoder, StandardScaler
from sklearn.model_selection      import train_test_split
from sklearn.metrics              import (accuracy_score, precision_score,
                                          recall_score, f1_score,
                                          confusion_matrix, classification_report,
                                          mean_absolute_error, mean_squared_error,
                                          r2_score)
from sklearn.linear_model         import LinearRegression, LogisticRegression
from sklearn.tree                 import DecisionTreeClassifier, plot_tree
from sklearn.svm                  import SVC
from sklearn.ensemble             import RandomForestClassifier
from sklearn.inspection           import DecisionBoundaryDisplay

# ──────────────────────────────────────────────────────────────
#  LOAD DATA
# ──────────────────────────────────────────────────────────────

df = pd.read_csv('loanData.csv')
print(df.info())

# ──────────────────────────────────────────────────────────────
#  STEP 1 — ENCODE
#  Order matters: Encode → Split → Scale
# ──────────────────────────────────────────────────────────────

# Drop ID — not a useful feature
df = df.drop(columns=['ApplicantID'])

# Nominal columns (no order) → get_dummies
# drop_first=True avoids dummy variable trap
df = pd.get_dummies(df, columns=['Employment_Status'], drop_first=True)
df = pd.get_dummies(df, columns=['Marital_Status'],    drop_first=True)

# Ordinal column (Month-to-Month < One Year < Two Year) → LabelEncoder
le = LabelEncoder()
df['Loan_Term'] = le.fit_transform(df['Loan_Term'])

print(df.head())  # sanity check

# ──────────────────────────────────────────────────────────────
#  STEP 2 — SPLIT
# ──────────────────────────────────────────────────────────────

X = df.drop(columns=['Approved'])
y = df['Approved']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42   # fixed seed → same split every run
)

# ──────────────────────────────────────────────────────────────
#  STEP 3 — SCALE
#  fit_transform on train (learn + apply)
#  transform only on test (apply only — no data leakage)
# ──────────────────────────────────────────────────────────────

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# New applicant — built once, reused across all models
# Columns must match exactly after encoding (check X.columns.tolist())
new_applicant = pd.DataFrame([{
    'Income':                          50000,
    'Credit_Score':                    710,
    'Loan_Amount':                     12000,
    'Existing_Debt':                   2000,
    'Loan_Term':                       1,     # 1 = One Year (label encoded)
    'Employment_Status_Self-Employed': 0,
    'Employment_Status_Unemployed':    0,     # 0,0 → Employed (dropped category)
    'Marital_Status_Married':          1,
    'Marital_Status_Single':           0
}])
new_applicant_scaled = scaler.transform(new_applicant)

# ══════════════════════════════════════════════════════════════
#  LINEAR REGRESSION
#  Use when: target is continuous (predicting a number)
#  Metrics:  MAE, RMSE, R²
# ══════════════════════════════════════════════════════════════

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred_lr)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2   = r2_score(y_test, y_pred_lr)

print("\n── Linear Regression ──")
print(f"MAE:  {mae:.2f}")   # avg error in same unit as target
print(f"RMSE: {rmse:.2f}")  # punishes big errors more (sensitive to outliers)
print(f"R²:   {r2:.2f}")    # % of variance explained, closer to 1 is better

print(f"Predicted Score: {lr_model.predict(new_applicant_scaled)[0]:.2f}")

# ══════════════════════════════════════════════════════════════
#  LOGISTIC REGRESSION
#  Use when: target is categorical (binary 0/1)
#  Metrics:  Accuracy, Precision, Recall, F1, Confusion Matrix
# ══════════════════════════════════════════════════════════════

log_model = LogisticRegression()
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)  # actual probabilities per class

print("\n── Logistic Regression ──")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_log):.2f}")   # overall correctness
print(f"Precision: {precision_score(y_test, y_pred_log):.2f}")  # when u predict +ve, how right are you
print(f"Recall:    {recall_score(y_test, y_pred_log):.2f}")     # out of all actual +ves, how many caught
print(f"F1 Score:  {f1_score(y_test, y_pred_log):.2f}")         # balance of precision & recall

cm = confusion_matrix(y_test, y_pred_log)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Logistic Regression — Confusion Matrix')
plt.show()

print(f"Loan Approved:       {log_model.predict(new_applicant_scaled)[0]}")
print(f"Approval Probability:{log_model.predict_proba(new_applicant_scaled)[0][1]:.2f}")

# ══════════════════════════════════════════════════════════════
#  DECISION TREE
#  Use when: classification or regression, need interpretability
#  Extras:   feature importance, tree visualization
#  Watch out: overfits by default → fix with max_depth
# ══════════════════════════════════════════════════════════════

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

print("\n── Decision Tree (no depth limit) ──")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.2f}")
print(classification_report(y_test, y_pred_dt))  # precision, recall, F1 all in one

# Feature importance — which features mattered most
importance = pd.Series(
    dt_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)
print(importance)

importance.plot(kind='bar')
plt.title('Decision Tree — Feature Importance')
plt.tight_layout()
plt.show()

# Visualize the tree
plt.figure(figsize=(12, 6))
plot_tree(dt_model, feature_names=X.columns,
          class_names=['Rejected', 'Approved'],
          filled=True, max_depth=3)
plt.title('Decision Tree — Visualization')
plt.show()

# Fix overfitting with max_depth
dt_model_pruned = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model_pruned.fit(X_train, y_train)
y_pred_dt_pruned = dt_model_pruned.predict(X_test)

print("\n── Decision Tree (max_depth=4) ──")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt_pruned):.2f}")

# ══════════════════════════════════════════════════════════════
#  SUPPORT VECTOR MACHINE (SVM)
#  Use when: classification, works well on smaller datasets
#  Key idea: finds hyperplane with maximum margin between classes
#  Extras:   decision boundary visualization (2 features only)
# ══════════════════════════════════════════════════════════════
#
#  Parameters:
#  kernel='rbf'    → non-linear boundary (use when data isn't linearly separable)
#  kernel='linear' → straight line boundary
#  C               → strictness. High C = less margin, risks overfitting
#  gamma           → influence range. High gamma = only nearby points matter
#  probability=True → needed to use predict_proba()

svm_model = SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)

print("\n── SVM ──")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.2f}")
print(classification_report(y_test, y_pred_svm))

cm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('SVM — Confusion Matrix')
plt.show()

print(f"Loan Approved:        {svm_model.predict(new_applicant_scaled)[0]}")
print(f"Approval Probability: {svm_model.predict_proba(new_applicant_scaled)[0][1]:.2f}")

# Decision boundary — only possible with 2 features (can't visualize N dimensions)
X_vis   = X_train[:, :2]
svm_vis = SVC(kernel='rbf', C=1, gamma='scale')
svm_vis.fit(X_vis, y_train)

DecisionBoundaryDisplay.from_estimator(
    svm_vis, X_vis,
    response_method='predict',
    cmap='coolwarm', alpha=0.5
)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_train, cmap='coolwarm', edgecolors='k')
plt.title('SVM — Decision Boundary (2 features)')
plt.show()

# ══════════════════════════════════════════════════════════════
#  RANDOM FOREST
#  Use when: classification, want better accuracy than one tree
#  Key idea: builds 100 trees, each on random subset → majority vote
#  Benefit:  less overfitting than single Decision Tree
# ══════════════════════════════════════════════════════════════

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\n── Random Forest ──")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
print(classification_report(y_test, y_pred_rf))

importance_rf = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

importance_rf.plot(kind='bar')
plt.title('Random Forest — Feature Importance')
plt.tight_layout()
plt.show()