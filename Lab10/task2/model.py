import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('loanData.csv')
TEST_SIZE=0.2
print(df.info())
print(df.head())

# drop unnecessary data
df = df.drop(columns='loan_id')
df.columns = df.columns.str.strip()
df = df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)

print(df['education'].unique().tolist())
print(df['self_employed'].unique().tolist())

# encode
le = LabelEncoder()
df['education'] = le.fit_transform(df['education'])
df['self_employed'] = le.fit_transform(df['self_employed'])
df['loan_status'] = le.fit_transform(df['loan_status'])
print(df.info())

# split
X = df.drop(columns='loan_status')
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=42
)

# scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X_train, y_train)

feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'Score': selector.scores_
}).sort_values(by='Score', ascending=False)

print(feature_scores)

feature_scores.plot(kind='bar', x='Feature', y='Score')
plt.title('Feature Selection via Importance Scores')
plt.tight_layout()
plt.show()

selector = SelectKBest(score_func=f_classif, k=8)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

# train
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

lr_model = LogisticRegression()
dt_model = DecisionTreeClassifier(max_depth=4)

lr_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)

# predit
lr_y_pred = lr_model.predict(X_test)
dt_y_pred = dt_model.predict(X_test)

# accuracy
print("Logistic Regression Accuracy")
print(f"Accuracy Score: {accuracy_score(y_test, lr_y_pred)}")
print(f"Precision: {precision_score(y_test, lr_y_pred)}")
print(f"Recall Score: {recall_score(y_test, lr_y_pred)}")
print(f"F1 Score: {f1_score(y_test, lr_y_pred)}")
print("\nDecision Tree Accuracy")
print(f"Accuracy Score: {accuracy_score(y_test, dt_y_pred)}")
print(f"Classification Report :\n{classification_report(y_test, dt_y_pred)}")

# predict on new data
new_applicant = pd.DataFrame([{
    "no_of_dependents": 4,
    "education": 0,
    "self_employed": 1,
    "income_annum": 7800000,
    "loan_amount": 7987987,
    "loan_term": 14,
    "cibil_score": 760,
    "residential_assets_value": 48000000,
    "commercial_assets_value": 9800000,
    "luxury_assets_value": 4213912,
    "bank_asset_value": 4329816,
}])

new_applicant_scaled = scaler.transform(new_applicant)
new_applicant_scaled = selector.transform(new_applicant_scaled)

lr_prediction = lr_model.predict(new_applicant_scaled)[0]
dt_prediction = dt_model.predict(new_applicant_scaled)[0]

print(f"LR Prediction: {lr_prediction}")
print(f"DT Prediction: {dt_prediction}")


