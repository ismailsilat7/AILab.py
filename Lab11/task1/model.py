import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Mall_Customers.csv')
print(df.head())
print(df.info())

df = df.drop(columns=['CustomerID'])
X = df.copy()

from sklearn.preprocessing import StandardScaler

X = pd.get_dummies(X, columns=['Gender'], drop_first=True)

scaler = StandardScaler()
cols_to_scale = [col for col in X.columns if col != 'Age']
X_scaled = X.copy()
X_scaled[cols_to_scale] = scaler.fit_transform(X_scaled[cols_to_scale])

from sklearn.cluster import KMeans

wcws_scaled = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcws_scaled.append(kmeans.inertia_)

plt.plot(range(1,11), wcws_scaled, marker='o')
plt.title('Elbow Method - Scaled')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.show()

kmeans_scaled = KMeans(n_clusters=6, init='k-means++', random_state=42)
y_pred_scaled = kmeans_scaled.fit_predict(X_scaled)

df['Cluster'] = y_pred_scaled
print(df.head())

plt.figure(figsize=(8,10))
sns.scatterplot(
    data=df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='Set1',
    s=100
)
plt.title('Customer Segments - Scaled')
plt.show()


wcws = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcws.append(kmeans.inertia_)

plt.plot(range(1,11), wcws, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=6, init='k-means++', random_state=42)
y_pred = kmeans.fit_predict(X)

df['Cluster_Unscaled'] = y_pred
print(df.head())

plt.figure(figsize=(8,10))
#  Plot unscaled
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Cluster_Unscaled', palette='Set1', s=100)
plt.title('Customer Segments — Unscaled')
plt.show()