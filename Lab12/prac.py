import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('telecom_churn.csv')

print("First 5 rows")
print(df.head())
print("Last 5 rows")
print(df.tail())
print("Rows 12-14")
print(df[12:15])
print("Last row")
print(df[-1:])

# (rows, cols)
print(df.shape)
# column names
print(df.columns.to_list())
# data types + non-null counts
print(df.info()) # structural information about the DataFrame
# describe
print(df.describe()) # statistical information 



print(df.isnull().sum()) # missing values per column
print(df.isnull().any()) # True/False — does this column have ANY nulls?

df['total day charge'].mean() # mean of one column
df[df['churn'] == 1].select_dtypes(include=['float64','int64']).mean() # mean of ALL numeric columns for churned users only

print(f"average total day minutes for users who churned: {df[df['churn'] == 1]['total day minutes'].mean()}") # df  [condition]  [column name]  .function()

# each condition must be fully wrapped in its own () including the comparison
print(f"maximum total international minutes for users who did churn AND have an international plan: {df[(df['churn'] == 1) & (df['international plan'] == 'yes')]['total intl minutes'].max()}")

# Sorting — sort_values()
# sort by one column, descending
print(f"\nsort by one column, descending\n{df.sort_values(by='total day charge', ascending=False).head()}")
# sort by multiple columns
print(f"\nsort by multiple columns\n{df.sort_values(by=['churn', 'total day charge'], ascending=[True, False]).head()}")

print(f"sort the dataframe by total night calls in ascending order\n{df.sort_values(by='total night calls', ascending=True).head()}")

# loc vs iloc
# loc — rows 0 to 5, columns 'State' to 'Area code' (INCLUSIVE on both ends)
# df.loc[0:5, 'State':'Area code']
# iloc — first 4 rows, first 4 columns (end is EXCLUSIVE, like normal Python slicing)
df.iloc[0:4, 0:4]

# df.loc[0:3, 'State':'Churn']
df.iloc[0:3, 0:2]

# apply, lambda, map
# apply: df.apply(lambda x: x.max() - x.min())
df.apply(np.max) #np.max passed - applies to each row

# lambda is ananoymous 1 line function
# normal function
def first_letter(state):
    return state[0]
# same thing as a lambda
lambda state: state[0]

# apply combined with lambda
# get all states starting with 'W'
df[df['state'].apply(lambda state: state[0] == 'W')].head()

# map(): replace values in a column using a dictionary
d = {'no': False, 'yes': True}
df['international plan'] = df['international plan'].map(d)
# Same thing can be done with replace()
df = df.replace({'voice mail plan': d})

# groupby()
# df.groupby(by='column')[columns_to_show].function()

# describe total day/eve/night minutes, grouped by churn
columns_to_show = ['total day minutes', 'total eve minutes', 'total night minutes']
df.groupby(['churn'])[columns_to_show].describe(percentiles=[]) # When you pass percentiles=[] you're saying "show no percentiles"

# multiple functions at once using agg()
df.groupby(['churn'])[columns_to_show].agg([np.mean, np.std, np.min, np.max]) # agg() lets you pass a list of functions to apply all at once

# Group by 'churn' and get the mean of 'total day minutes' and 'total night minutes'
df.groupby('churn')[['total day minutes', 'total night minutes']].mean()
# Same grouping but get both mean and max using agg()
df.groupby('churn')[['total day minutes', 'total night minutes']].agg(['mean', 'max']) # can pass the func names as strings. Strings like 'mean', 'max', 'min', 'std' are all valid inside agg()

"""
So you have 3 equivalent styles for agg():
.agg([np.mean, np.max])       # numpy functions
.agg(['mean', 'max'])         # strings ← cleanest
.agg([np.mean, 'max'])        # mix (also valid)
"""

# crosstab - contingency tables - how two categorical variables relate to each other.
print(pd.crosstab(df['churn'], df['international plan'], margins=True)) # gives you a frequency table — how many users fall into each combination. 
# margins=True adds a totals row/column
# normalize=True gives you proportions instead of counts

# pivot table - More Flexible Summary
print(df.pivot_table(['total day calls', 'total eve calls', 'total night calls'], ['area code'], aggfunc='mean'))
"""
first argument → columns to calculate stats for
second argument → column to group by (becomes the index)
aggfunc → what stat to calculate: 'mean', 'sum', 'max', 'min', 'std'
"""

pd.crosstab(df['churn'], df['international plan'], margins=True)
df.pivot_table(['total day calls'], ['area code'], aggfunc='std')

# Adding & Dropping Columns
# method 1 - insert()
total_calls = df['total day calls'] + df['total night calls'] + df['total eve calls'] + df['total intl calls']
df.insert(loc=len(df.columns), column='total calls', value=total_calls) # loc=len(df.columns) pastes it at the very end

# method 2 - direct assignment
df['total charge'] = df['total day charge'] + df['total eve charge'] + df['total night charge'] + df['total intl charge']

# Dropping columns or rows
# drop columns
df.drop(['total charge'], axis=1, inplace=True)

# drop rows by index
# df.drop([1, 2]).head()
df['total minutes'] = df['total day minutes'] + df['total night minutes']
df.drop(['total minutes'], axis = 1, inplace=True)

# Correlation + Visualizations
# Step 1 — Find Correlated Features
df_num = df.select_dtypes(include=['float64', 'int64']) #filter numeric cols only
df_num_corr = df_num.corr()
print(df_num_corr)

# Step 2 - Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df_num_corr, annot=True)
plt.show()

# Step 3 - Histogram for one column
plt.figure(figsize=(9,8))
sns.histplot(df['total day minutes'], color='g', bins=100, kde=True)
plt.show()

# Step 4 - Histogram for all numeric columns
df_num.hist(figsize=(16,20), bins=50, xlabelsize=8, ylabelsize=8)
plt.show()



"""
# Matplotlib (plt) — The Foundation
import matplotlib.pyplot as plt

plt.figure(figsize=(width, height))  # create canvas & set size
# ... plotting code goes here ...
plt.show() # display it
"""

"""
# Seaborn (sns) - Built on top of plt
import seaborn as sns

# The three plots you need for your exam:
# Countplot (barchart for categories)
sns.countplot(x='international plan', hue='churn', data=df)
plt.show()
# x= → column on x-axis
# hue= → split bars by this column (adds color grouping)
# data= → your dataframe

# Histplot (distribution of numeric column)
plt.figure(figsize=(9, 8))
sns.histplot(df['total day minutes'], color='g', bins=100, kde=True)
plt.show()
# bins= → number of bars
# kde=True → draws smooth curve on top
# color= → bar color

# Heatmap (correlation matrix)
plt.figure(figsize=(12, 8))
sns.heatmap(df_num.corr(), annot=True)
plt.show()
# annot=True → shows correlation numbers inside cells
"""

plt.figure(figsize=(9, 8)) # this works fine without plt.figure()
sns.countplot(x='customer service calls', hue='churn', data=df)
plt.show()

df_num = df.select_dtypes(include=['int64', 'float64'])
df_num_corr = df_num.corr()
plt.figure(figsize=(16, 10))
sns.heatmap(df_num_corr, annot=True)
plt.show()

"""
The three visualization patterns to remember for your exam:
# countplot
sns.countplot(x='column', hue='split_by', data=df)

# histplot
plt.figure(figsize=(9, 8))
sns.histplot(df['column'], bins=100, kde=True)

# heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_num.corr(), annot=True)
"""

# Cleaning flow
df.info()                          # 1. spot missing values & wrong types
df.isnull().sum()                  # 2. count nulls per column
df['col'] = df['col'].astype('int64')  # 3. fix types
df2 = df[[col for col in df if df[col].count() / len(df) >= 0.3]]  # 4. drop sparse columns
del df2['Id']                      # 5. drop irrelevant columns
df = df2

