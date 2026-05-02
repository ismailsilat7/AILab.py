import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# pd.set_option('display.max_rows', None)

data = pd.read_csv("house_prices_practice.csv")

# 1. Understanding data
print(data.head()) # first 5 rows
print(data.tail()) # last 5 rows
print(data.shape)
print(data.describe())
print(data.columns.to_list())
# checking unique values
print(data.nunique())
print(data['SaleType'].unique())

print(data.info())

df_num = data.select_dtypes(include=['int64', 'float64'])
print(f"Numerical: {df_num.columns.to_list()}")
print()
df_cat = data.select_dtypes(include=['object', 'str'])
print(f"Categorical: {df_cat.columns.to_list()}")


# 2. Cleaning
missing = data.isnull().sum()
missing_pct = (missing/len(data)) * 100
missing_pct_data = pd.DataFrame({'count': missing, 'pct': missing_pct})
print(missing_pct_data[missing_pct_data['count'] > 0].sort_values('pct', ascending=False))
# drop columns with more than 50% of the data missing
cols_to_drop = missing_pct[missing_pct > 0.5 * 100].index
data.drop(columns=cols_to_drop, inplace=True)
# filling numerical with median & categorical with mode
for col in data.select_dtypes(include=['int64', 'float64']):
    data[col] = data[col].fillna(data[col].median())
for col in data.select_dtypes(include=['object', 'str']):
    data[col] = data[col].fillna(data[col].mode()[0])
print(f"\nMissing values count after cleaning: {data.isnull().sum().sum()}")


# 3. Relationship Analysis

# correlation
corelation = data.select_dtypes(include=['int64', 'float64']).corr()

plt.figure(figsize=(20, 16))
sns.heatmap(corelation, xticklabels=corelation.columns, yticklabels=corelation.columns, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.1f')
plt.title('Correlation Heatmap')
plt.show()

#historgram
print()
plt.figure(figsize=(9,8))
sns.histplot(data['SalePrice'], color='g', bins=100, kde=True)
plt.title('Distribution of SalePrice')
plt.show()
print()
plt.figure(figsize=(9,8))
sns.histplot(data['GrLivArea'], color='g', bins=200, kde=True)
plt.title('Distribution of GrLivArea')
plt.show()

#histogram for all numerical columns
df_num = data.select_dtypes(include=['int64', 'float64'])
df_num.hist(figsize=(16,20), bins=50, xlabelsize=8, ylabelsize=8)
plt.tight_layout()
plt.show()


