Exploratory Data Analysis
Exploratory data analysis (EDA)  
• Exploratory data analysis is an approach for analyzing data sets to summarize  their main characteristics, often with visual methods 
• EDA is statisticians way of story telling where you explore data, find patterns  and tells insights
Exploratory data analysis (EDA) 
• EDA usually employs graphical techniques to get better understanding of  data, such as Histograms, Line charts, Box plots etc. 
• However, some other techniques like tabulation, clustering, simple model building can  also be used 
• Often undetected errors in data are discovered in this phase 

Exploratory data analysis (EDA) 
• Overlay plots is also a useful EDA technique, such as Pareto diagram • Overlay Plot allows one or more new plot curve(s) to be drawn over existing plots  using data from a Parametric, Lookup, Arrays, or Integral table. 

Exploratory data analysis (EDA) 
• Brushing and Linking (IVA) 


Pandas Library 
• Pandas is a Python library that provides extensive means for data analysis 
• Data scientists often work with data stored in table formats like .csv, .tsv, or  .xlsx 
• Pandas makes it very convenient to load, process, and analyze such tabular  data using SQL-like queries 
• In conjunction with Matplotlib and Seaborn, Pandas provides a wide range of  opportunities for visual analysis of tabular data
Pandas Library 
• The main data structures in Pandas are implemented with Series and  DataFrame classes 
• Series is a one-dimensional indexed array of some fixed data type  
• DataFrame is a two-dimensional data structure - a table - where each column  contains data of the same type 
• It as a dictionary of Series instances 
• Rows correspond to instances (examples, observations, etc.), and  
• Columns correspond to features of these instances.
Importing Libraries 
Code: 
import numpy as np 
import pandas as pd 
# we don't like warnings 
# you can comment the following 2 lines if you'd like to 
import warnings 
warnings.filterwarnings('ignore')
Analyzing a dataset on the churn rate of telecom operator clients • Read the data (using read_csv), and take a look at the first 5 lines using the head method 

Code: 
df = pd.read_csv('Desktop/Python Tutorial/telecom_churn.csv') df.head() 
Each row corresponds to one client, an  instance, and columns are features of this  instance.



Retrieving data 
If we need the first or the last line of the data frame, we can use the df[:1] or df[- 1:] construct: 
Code: 
df[12:15] 
df[-1:]

Data dimensionality, feature names, and feature types 
Code: 
Data Dimensionality: 
print(df.shape) 
Printing out column names using columns: 
print(df.columns)


Feature types: 
Use the info() method to output some general  information about the dataframe. 
Code: 
print(df.info())
• bool, int64, float64 and object are the data types  of our features 
• With this same method, we can easily see if  there are any missing values 

Changing the Feature Type • Change the column type with the astype method  
The number of individuals or items moving out of a collective  group over a specific period.
• Let’s apply this method to the churn feature to convert it into  int64 
• The describe method shows basic statistical characteristics of  each numerical feature (int64 and float64 types) 
• number of non-missing values, mean, standard deviation, range, median,  0.25 and 0.75 quartiles 
Code: 
df['churn'] = df['churn'].astype('int64’) df.describe() 



Statistics of non-numerical features 
Explicitly indicate data types of interest in the include parameter 
Code: 
df.describe(include=['object', 'bool'])
Statistics of Categorical (type object) and Boolean (type bool) features 
• For categorical (type object) and boolean (type bool) features we can use the  value_counts method 
• Let’s have a look at the distribution of Churn 
Code: 
df['churn'].value_counts() 
df['churn'].value_counts(normalize=True)

DataFrame Sorting 
• A DataFrame can be sorted by the value of one of the variables (i.e columns).  
• For example, we can sort by Total day charge (use ascending=False to sort in descending  order) Code: 
df.sort_values(by=‘total day charge', ascending=False).head()

Sort by multiple columns: 
Code: 
df.sort_values(by=[‘churn', ‘total day charge'], 
ascending=[True, False]).head()


Single Column Statistics: Indexing and retrieving data 
(a) To get a single column, we can use a DataFrame['Name'] construction  What is the proportion of churned users in our dataframe? 
Code: 
df[‘churn'].mean()
Single Column Statistics: Indexing and retrieving data (b) Boolean indexing with one column:  
• The syntax is df[P(df['Name'])], where P is some logical condition  that is checked for each element of the Name column.  
• The result of such indexing is the DataFrame consisting only of  rows that satisfy the P condition on the Name column. 
What are average values of numerical features for churned users? 
Code: 
df[df[‘churn'] == 1].mean()
Multiple Column Statistics: Indexing and retrieving data 

How much time (on average) do churned users spend  on the phone during daytime? 
Code: 
df[df['churn'] == 1]['total day minutes'].mean() 

What is the maximum length of international calls among  loyal users (Churn == 0) who do not have an international  plan?
Code: 
df[(df['churn'] == 1) & (df['international plan']== 'yes')]['total intl minutes'].max() 
Indexing and retrieving data 
(c) DataFrames indexing by column name (label) or row name (index) or by the serial  number of a row 
• The loc method is used for indexing by name, while iloc() is used for indexing by number • In the first case below, we say "give us the values of the rows with index from 0 to 5 (inclusive) and  columns labeled from State to Area code (inclusive)" 
• In the second case, we say "give us the values of the first five rows in the first three columns" (as in a  typical Python slice: the maximal value is not included)
Code: 
df.loc[0:5, 'State':'Area code’] 
df.iloc[0:4, 0:4] 
Indexing and retrieving data 
(d) If we need the first or the last line of the data frame, we can use the df[:1] or  df[-1:] construct: 
Code: 
df[12:15] # Index 12 to 15 
df[-1:]
Applying Functions to Cells, Columns and Rows 
To apply functions to each column, use apply(): 
The apply method can also be used to apply a function to each  
row.  
• To do this, specify axis=1 
Code: 
df.apply(np.max)  
How to a particular Row? 
Home Task
Applying Functions to Cells, Columns and Rows 
Lambda function:  
If we need to select all states starting with W, we can do it like this: 

Code: 
df[df[‘state'].apply(lambda state: state[0] == 'W')].head() 
A lambda function is a small  anonymous function. 
A lambda function can take any  number of arguments, but can only  have one expression. 
x = lambda a, b : a * b 
print(x(5, 6)) 

Applying Functions to Cells, Columns and Rows
Map function:  
• The map method can be used to replace values in a column by passing a dictionary of the  form {old_value: new_value} as its argument: 
Code: 
d = {'No' : False, 'Yes' : True} 
df['International plan'] = df['International plan'].map(d) 
df.head() 
The same thing can be done with the replace  
method: 
df = df.replace({'Voice mail plan': d}) 
df.head() 
Grouping Data df.groupby(by=grouping_columns)[columns_to_show].function() 
• First, the groupby method divides the grouping_columns by their values.  • They become a new index in the resulting dataframe. 
• Then, columns of interest are selected (columns_to_show).  
• If columns_to_show is not included, all non groupby clauses will be included. 
• Finally, one or several functions are applied to the obtained groups per selected columns. 
Code: 
columns_to_show = ['total day minutes', 'total eve minutes’, total night minutes'] 
df.groupby(['churn'])[columns_to_show].describe(percentiles=[])

Grouping Data
Passing a list of functions to agg(): 
Code: 
columns_to_show = ['total day minutes', 'total eve minutes',  
'total night minutes'] 
df.groupby(['churn'])[columns_to_show].agg([np.mean, np.std, np.min, np.max]) 

Summary tables 
• Suppose we want to see how the observations in our sample are distributed in the context  of two variables - Churn and International plan 
• To do so, we can build a contingency table using the crosstab method 
Code: 
pd.crosstab(df['Churn'], df['International plan']) 
pd.crosstab(df['Churn'], df['Voice mail plan'], normalize=True)
Pivot tables  Pivot_table method takes the following parameters: 
• values – a list of variables to calculate statistics for 
• index – a list of variables to group data by 
• aggfunc – what statistics we need to calculate for groups: • sum, mean, maximum, minimum or something else 
Let’s look at the average number of day, evening, and night calls by area code: Code: 
df.pivot_table(['total day calls', 'total eve calls', 'total night  
calls'], ['area code'], aggfunc='std’)  
#sum , mean, maximum, minimum or something else.
DataFrame transformations 
Adding columns to a DataFrame: 
For example: Calculate the total number of calls for all users • Create the total_calls Series and paste it into the DataFrame 
Code: 
total_calls = df['Total day calls'] + df['Total eve calls'] + \ 
df['Total night calls'] + df['Total intl calls'] 
df.insert(loc=len(df.columns), column='Total calls', value=total_calls)  df.head() 
# loc parameter is the number of columns after which to insert the Series object # we set it to len(df.columns) to paste it at the very end of the dataframe

DataFrame transformations 
Adding columns to a DataFrame:
• Add a column without creating an intermediate Series instance 
Code: 
df['Total charge'] = df['total day charge'] + df['total eve charge'] + \ df['total night charge'] + df['total intl charge'] 
df.head() 


DataFrame transformations 
Delete columns or rows
• Use the drop method, passing the required indexes and the axis parameter (1 if you delete  columns, and nothing or 0 if you delete rows) 
• The inplace argument tells whether to change the original DataFrame.  • With inplace=False, the drop method doesn't change the existing DataFrame and returns a new one with  dropped rows or columns.  
• With inplace=True, it alters the DataFrame. 
Code: 
# get rid of just created columns 
df.drop(['Total charge', 'Total calls'], axis=1, inplace=True)  
# and here’s how you can delete rows 
df.drop([1, 2]).head()  
Predicting telecom churn Using a crosstab contingency table  
How churn rate is related to the International plan feature.  • Using a crosstab contingency table and  
• Also through visual analysis with Seaborn 
Code: 
pd.crosstab(df['churn'], df['international plan'], margins=True)
Predicting telecom churn Visual  analysis with Seaborn 
Code: 
# some imports to set up plotting  
import matplotlib.pyplot as plt 
# pip install seaborn  
import seaborn as sns 
sns.countplot(x='international plan', hue='churn', data=df);
Predicting telecom churn Using a crosstab contingency table  Code: 
pd.crosstab(df['churn'], df['customer service calls'], margins=True)


Predicting telecom churn Visual analysis with Seaborn 
Code: 
sns.countplot(x='Customer service calls', hue='Churn', data=df);


Predicting telecom churn Using a crosstab contingency table 
Code: 
df['Many_service_calls'] = (df['customer service calls'] > 3).astype('int') 
pd.crosstab(df['Many_service_calls'], df['churn'], margins=True) 

Predicting telecom churn Visual analysis with Seaborn
Code: 
sns.countplot(x='Many_service_calls', hue='churn', data=df); 


Predicting telecom churn Using a crosstab contingency table 
• Contingency table that relates Churn with both International plan and freshly created  Many_service_calls. 
Code: 
pd.crosstab(df['Many_service_calls'] & df['international plan'] , df['churn']) 

EDA on another Dataset
Preparations 
For the preparation, lets first import the necessary libraries and load the files needed for our  EDA 
Code: 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import warnings 
warnings.filterwarnings('ignore') 
# Comment this if the data visualisations doesn't work on your side 
%matplotlib inline 
plt.style.use('bmh')
Preparations 
Code: 
df = pd.read_csv('train.csv') 
df.head() 
df.info()
• Some features won't be relevant in our exploratory  analysis as there are too much missing values  (such as Alley and PoolQC) 
• Better to concentrate on the ones which can give  us real insights 
Preparations 
• Remove Id and  
• Features with 30% or less NaN 
values 
Code: 
# df.count() does not include NaN values 
df2 = df[[column for column in df if df[column].count() / len(df) >= 0.3]] 
del df2['Id'] 
print("List of dropped columns:", end=" ") 
for c in df.columns: 
if c not in df2.columns: 
print(c, end=", ") 
print('\n') 
df = df2
How the housing price is distributed 
Code: 
print(df['SalePrice'].describe()) 
plt.figure(figsize=(9, 8)) 
sns.distplot(df['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.4}); 
• Prices are skewed right and some outliers lies above  ~500,000  
• We will eventually want to get rid of the them to get a  normal distribution of the independent variable (`SalePrice`) 
How? Home Task
Listing Data from a dataset 
List all the types of our data from our dataset and take only the numerical ones: 
Code: 
list(set(df.dtypes.tolist())) 
df_num = df.select_dtypes(include = ['float64', 'int64']) 
df_num.head()

Code: 
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);  
# ; avoid having the matplotlib verbose informations


Correlation 
Features such as `1stFlrSF`, `TotalBsmtSF`, `LotFrontage`,  
`GrLiveArea`... seems to share a similar distribution to the  
one we have with `SalePrice` 
• Find which features are strongly correlated with SalePrice 
• We'll store them in a var called golden_features_list 
• We'll reuse our df_num dataset to do so
Code: 
df_num_corr = df_num.corr()['SalePrice'][:-1] # -1 because the latest row is SalePrice 
golden_features_list = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending=False) 
print("There is {} strongly correlated values with SalePrice:\n{}".format(len(golden_features_list),  
golden_features_list)) 
