import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('studentData.csv')
TEST_SIZE = 0.2

print(df.info())
print(df.head())

# drop any useless columns 
# none

# encode categoricals correctly
le = LabelEncoder()
df['Extracurricular Activities'] = le.fit_transform(df['Extracurricular Activities'])

# split

X = df.drop(columns='Performance Index')
y = df['Performance Index']
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

# train model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# accuracy
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMAE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R2_score: {r2_score(y_test, y_pred):.2f}")

# predict for new_student
new_student = pd.DataFrame([{
    "Hours Studied": 7,
    "Previous Scores": 80,
    "Extracurricular Activities": 1,
    "Sleep Hours": 7,
    "Sample Question Papers Practiced": 5
}])

new_student_scaled = scaler.transform(new_student)

prediction = model.predict(new_student_scaled)
print(f"Prediction (Performance Index): {prediction[0]:.2f}")



