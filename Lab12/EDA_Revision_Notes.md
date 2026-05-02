# EDA Revision Notes
### Based on Telecom Churn Dataset

---

## 0. Imports & Setup

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline  # Jupyter only
```

---

## 1. Loading & Previewing Data

```python
df = pd.read_csv('telecom_churn.csv')

df.head()       # first 5 rows
df.tail()       # last 5 rows
df[12:15]       # rows 12 to 14 (slicing)
df[-1:]         # last row
```

---

## 2. Understanding the Dataset

```python
print(df.shape)    # (rows, columns)
print(df.columns.to_list())  # column names
print(df.info())   # data types + non-null counts  ← structural
print(df.describe())         # statistical summary  ← numerical
```

### `.info()` vs `.describe()`

| | `.info()` | `.describe()` |
|---|---|---|
| What it shows | Data types, non-null counts | Count, mean, std, min, max, quartiles |
| Best for | Spotting missing values & types | Understanding numerical distributions |

```python
# For non-numerical columns (object, bool):
df.describe(include=['object', 'bool'])  # shows count, unique, top, freq
```

### Missing Values

```python
df.isnull().sum()   # count of missing values per column
df.isnull().any()   # True/False — does this column have ANY nulls?
```

---

## 3. Boolean Indexing

**Pattern:** `df[condition][column].function()`

```python
# Single condition
df[df['churn'] == 1].mean()                        # mean of all numeric cols for churned users
df[df['churn'] == 1]['total day minutes'].mean()   # mean of one column for churned users

# Multiple conditions — each condition must be wrapped in ()
df[(df['churn'] == 1) & (df['international plan'] == 'yes')]['total intl minutes'].max()
```

> ⚠️ Each condition needs its own `()` including the comparison:
> ✅ `(df['col'] == value)` — NOT ~~`(df['col']) == value`~~

---

## 4. Sorting

```python
# Sort by one column, descending
df.sort_values(by='total day charge', ascending=False).head()

# Sort by multiple columns
df.sort_values(by=['churn', 'total day charge'], ascending=[True, False]).head()
```

---

## 5. `loc` vs `iloc`

```python
df.loc[0:5, 'State':'Area code']   # by label — end is INCLUSIVE
df.iloc[0:4, 0:4]                  # by number — end is EXCLUSIVE
```

| | End value |
|---|---|
| `loc` | Included |
| `iloc` | Excluded |

---

## 6. `apply()`, `lambda`, `map()`

### `apply()` — apply a function to each column (or row)

```python
df.apply(np.max)              # max of every column
df.apply(np.max, axis=1)      # max across each row
df.apply(lambda x: x.max() - x.min())  # custom function — range of each column
```

### `lambda` — small anonymous one-line function

```python
lambda state: state[0]   # returns first character of state

# Filter rows where state starts with 'W'
df[df['state'].apply(lambda state: state[0] == 'W')].head()
```

### `map()` — replace column values using a dictionary

```python
d = {'no': False, 'yes': True}
df['international plan'] = df['international plan'].map(d)

# map() is strict — unmatched values become NaN!
# Always check exact values first:
print(df['international plan'].unique())
```

### `replace()` — same as map but safer alternative

```python
df = df.replace({'voice mail plan': d})
```

---

## 7. `groupby()`

**Pattern:** `df.groupby(by='column')[columns_to_show].function()`

```python
columns_to_show = ['total day minutes', 'total eve minutes', 'total night minutes']

# Single function
df.groupby(['churn'])[columns_to_show].describe(percentiles=[])  # [] = no quartiles shown

# Multiple functions with agg()
df.groupby(['churn'])[columns_to_show].agg([np.mean, np.std, np.min, np.max])
```

### 3 equivalent styles for `agg()`:

```python
.agg([np.mean, np.max])    # numpy functions
.agg(['mean', 'max'])      # strings ← cleanest, no numpy needed
.agg([np.mean, 'max'])     # mix — also valid
```

---

## 8. `crosstab` & `pivot_table`

### `crosstab` — frequency table for two categorical variables

```python
pd.crosstab(df['churn'], df['international plan'])                   # basic
pd.crosstab(df['churn'], df['international plan'], margins=True)     # + totals row/col
pd.crosstab(df['churn'], df['voice mail plan'], normalize=True)      # proportions instead of counts
```

### `pivot_table` — flexible summary with stats

```python
df.pivot_table(
    ['total day calls', 'total eve calls', 'total night calls'],  # columns to calculate stats for
    ['area code'],                                                 # group by (becomes index)
    aggfunc='mean'                                                 # 'mean', 'sum', 'max', 'min', 'std'
)
```

| | `crosstab` | `pivot_table` |
|---|---|---|
| Best for | Counting category combinations | Calculating stats across groups |

---

## 9. Adding & Dropping Columns

### Adding

```python
# Method 1 — insert() (choose position)
total_calls = df['total day calls'] + df['total eve calls'] + \
              df['total night calls'] + df['total intl calls']
df.insert(loc=len(df.columns), column='total calls', value=total_calls)
# loc=len(df.columns) → paste at the very end

# Method 2 — direct assignment (simpler, always appends at end)
df['total charge'] = df['total day charge'] + df['total eve charge'] + \
                     df['total night charge'] + df['total intl charge']
```

### Dropping

```python
df.drop(['total charge', 'total calls'], axis=1, inplace=True)  # drop columns
df.drop([1, 2]).head()                                           # drop rows by index
```

> `axis=1` → columns | `axis=0` (default) → rows  
> `inplace=True` → modifies original DataFrame directly

---

## 10. Correlation

```python
# Step 1 — Select numeric columns only
df_num = df.select_dtypes(include=['float64', 'int64'])

# Step 2 — Compute correlation matrix
df_num_corr = df_num.corr()

# Step 3 — Find strongly correlated features (|corr| > 0.5)
golden_features = df_num_corr['target_column'][:-1]
golden_features = golden_features[abs(golden_features) > 0.5].sort_values(ascending=False)
print(golden_features)
```

---

## 11. Visualizations

### Heatmap (correlation matrix)

```python
df_num = df.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(12, 8))
sns.heatmap(df_num.corr(), annot=True)  # annot=True shows numbers inside cells
plt.show()
```

### Histogram — one column

```python
plt.figure(figsize=(9, 8))
sns.histplot(df['total day minutes'], color='g', bins=100, kde=True)
# kde=True → smooth curve on top
plt.show()
```

### Histogram — all numeric columns

```python
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
plt.show()
```

### Countplot (bar chart for categories)

```python
sns.countplot(x='customer service calls', hue='churn', data=df)
# x=    → column on x-axis
# hue=  → split bars by this column (color grouping)
# data= → your dataframe
plt.show()
```

---

## Quick Reference — Visualization Patterns

```python
# Countplot
sns.countplot(x='column', hue='split_by', data=df)

# Histplot
plt.figure(figsize=(9, 8))
sns.histplot(df['column'], bins=100, kde=True)

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_num.corr(), annot=True)
```

> Always end with `plt.show()`  
> `plt.figure(figsize=(w, h))` is optional for countplot but recommended for histplot and heatmap

---

## Common Gotchas

| Mistake | Fix |
|---|---|
| `(df['col']) == value` in multi-condition filter | Wrap fully: `(df['col'] == value)` |
| `map()` returns NaN | Check `df['col'].unique()` before mapping |
| `distplot` warning | Use `histplot` instead (distplot is deprecated) |
| `describe()` skips bool/object | Use `describe(include=['object', 'bool'])` |
| `loc` vs `iloc` end index | `loc` is inclusive, `iloc` is exclusive |
