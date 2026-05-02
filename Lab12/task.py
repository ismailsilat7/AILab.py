import pandas as pd
import numpy as np

# Data Handling

# Load & first look
df = pd.read_csv('house_prices_practice.csv')
print(df.head())
print()
print(df.shape)
print()
print(df.columns.tolist())
print()


# Numerical vs Categorical

# Numerical features (int64, float64)
print()
df_num = df.select_dtypes(include=['int64','float64'])
print("Numerical:", df_num.columns.tolist())
# Categorical features (object)
print()
df_cat = df.select_dtypes(include=['object', 'str'])
print("Categorical:", df_cat.columns.tolist())

# Basic Stats
print()
print(df.info())
print(df.describe())
print(df.describe(include=['object', 'str']))



# Data Cleaning

#  Find missing values
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({'count': missing, 'pct': missing_pct})
print(missing_df[missing_df['count'] > 0].sort_values('pct', ascending=False))

# Keep only columns with ≥50% non-null values
# df = df[[col for col in df if df[col].count()/len(df) >= 0.5]]
# Or explicitly:
threshold = 0.5
cols_to_drop = missing_pct[missing_pct > threshold * 100].index
df.drop(columns=cols_to_drop, inplace=True)

# Numerical:fill with median (robust to outliers)
for col in df.select_dtypes(include=['int64','float64']).columns:
    df[col] = df[col].fillna(df[col].median())

# Categorical: fill with mode
for col in df.select_dtypes(include=['object', 'str']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Verify no missing values remain
print(df.isnull().sum().sum())  # should be 0

# Univariate Analysis
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram of SalePrice
print()
plt.figure(figsize=(9,8))
sns.histplot(df['SalePrice'], color='g', bins=100, kde=True)
plt.title('Distribution of SalePrice')
plt.show()

# Histogram of GrLivArea
plt.figure(figsize=(9,8))
sns.histplot(df['GrLivArea'], bins=50, color='steelblue', kde=True)
plt.title('Distribution of GrLivArea')
plt.show()

# Histogram for ALL numerical columns
df_num = df.select_dtypes(include=['float64','int64'])
df_num.hist(figsize=(16,20), bins=50, xlabelsize=8, ylabelsize=8)
plt.tight_layout()
plt.show()

# Boxplot to detect outliers

# Single variable boxplot
plt.figure(figsize=(8,4))
sns.boxplot(x=df['SalePrice'])
plt.title('Boxplot of SalePrice')
plt.show()

# Side-by-side for comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(x=df['SalePrice'], ax=axes[0])
sns.boxplot(x=df['GrLivArea'], ax=axes[1])
plt.show()

# Multivariate Analysis / Heatmap
df_num = df.select_dtypes(include=['float64','int64'])
corr_matrix = df_num.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix,
            annot=True, # show numbers in cells
            fmt='.2f', # 2 decimal places
            cmap='coolwarm', # red=positive, blue=negative
            linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Feature Engineering

# HouseAge = year it was sold minus year it was built
df['HouseAge'] = df['YrSold'] - df['YearBuilt']
print(df[['YrSold', 'YearBuilt', 'HouseAge']].head())

# drop irrelevant feature
# Drop Id, it's just an index, not a feature
df.drop(['Id'], axis=1, inplace=True)

# Linear Regression evaluate with MAE, RMSE, R²
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Only use numerical columns for simplicity
df_num = df.select_dtypes(include=['float64','int64'])
df_num.dropna(inplace=True)

X = df_num.drop('SalePrice', axis=1)
y = df_num['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"MAE:  {mae:.2f}")   # avg absolute error in $
print(f"RMSE: {rmse:.2f}")  # penalizes large errors more
print(f"R²:   {r2:.4f}")    # 1.0 = perfect, 0 = no better than mean
