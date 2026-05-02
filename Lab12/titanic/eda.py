"""
Titanic Dataset - https://www.kaggle.com/competitions/titanic/data (download train.csv) and rename to data.csv

1. Import all necessary libraries
2. Load train.csv into df and preview the first 5 rows

Understanding the Data
3. Print the shape, column names, and data types/null counts
Statistics
4. Print summary statistics for numerical columns
5. Print value counts for the 'Survived' column (normalized)

Filtering
6. What is the average 'Age' of passengers who survived (Survived == 1)?
7. What is the maximum 'Fare' paid by passengers in 3rd class (Pclass == 3) who did NOT survive?

Sorting & Transformations
8. Sort by 'Fare' descending, show top 5
9. Add a new column 'FamilySize' = 'SibSp' + 'Parch'
10. Drop that column immediately after

Groupby
11. Group by 'Survived' and get mean and std of 'Age' and 'Fare' using agg()
Summary Tables
12. Build a crosstab of 'Survived' vs 'Pclass' with totals
13. Build a pivot table of average 'Fare' grouped by 'Pclass'

Visualizations
14. Plot a countplot of 'Pclass' split by 'Survived'
15. Plot a histogram of 'Age'
16. Plot a heatmap of correlations for all numeric columns
"""

# import libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load data
df = pd.read_csv("data.csv")

# understanding
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.to_list()}")
print(df.info())
print(f"Numerical columns: {df.select_dtypes(include=['int64', 'float64']).columns.to_list()}")
print(f"Categorical columns: {df.select_dtypes(include=['str', 'object']).columns.to_list()}")

# cleaning
print(df.isnull().sum())
df2 = df[[col for col in df if df[col].count()/len(df) >= 0.3]]
del df2['PassengerId']
df = df2
print(df.isnull().sum())
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
print(df.isnull().sum())

# statistics
print(df.describe())
print(f"Value counts for 'Survived': {df['Survived'].value_counts(normalize=True)}")

# filtering
print(f"Avergae age of passengers who survived: {df[(df['Survived'] == 1)]['Age'].mean()}")
print(f"Maximum fare paid by passenger who didnt survive: {df[((df['Pclass'] == 3) & (df['Survived'] == 0))]['Fare'].max()}")

# sorting & transformation
print(f"Sort by 'Fare' desc: \n{df.sort_values(by=['Fare'], ascending=False).head()}")
df['FamilySize'] = df['SibSp'] + df['Parch']
print(f"New column Family size: \n{df['FamilySize']}")
df.drop('FamilySize', axis=1, inplace=True)

# Groupby
print(df.groupby('Survived')[['Age', 'Fare']].agg(['mean', 'std']))

# Summary tables
print(df.pivot_table(['Fare'], ['Pclass'], aggfunc='mean'))
pd.crosstab(df['Survived'], df['Pclass'], normalize=True, margins=True)

# Visualizations
plt.figure(figsize=(8,10))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.show()

plt.figure(figsize=(8,10))
sns.histplot(df['Age'], color='g', bins=100, kde=True)
plt.show()

df.hist(figsize=(16,20), bins=50, xlabelsize=8, ylabelsize=8)
plt.show()

df_num = df.select_dtypes(include=['int64', 'float64'])
df_num_corr = df_num.corr()
print(df_num_corr)

plt.figure(figsize=(16, 20))
sns.heatmap(df_num_corr, annot=True)
plt.show()
