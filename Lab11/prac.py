
"""
# supervised pattern you know
model.fit(X_train, y_train)
model.predict(X_test)

# unsupervised pattern — combined, no train/test split
model.fit_predict(X)  # learns clusters + assigns each point in one shot
"""

# K-Means Clustering
# Use when: you want to group similar data points together without any labels

# WCSS sum of squared distances between each point and its cluster's centroid, low WCSS -> points closer, tight clusters, better

# Elbow Method for finding optimal K: tries K=1 to K=10, calculates WCSS for each, and plots it
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv('Mall_Customers.csv')
print(df.info())
print(df.head())

le = LabelEncoder()
df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

df = df.drop(columns=['CustomerID'])
X = df.copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
y_pred = kmeans.fit_predict(X_scaled)

df['Cluster'] = y_pred
print(df.head())

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='Set1',
    s=100
)
plt.title('Customer Segments')
plt.show()
