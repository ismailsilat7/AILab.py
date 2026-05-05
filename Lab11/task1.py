import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('Mall_Customers.csv')
x = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values


wcss_list = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss_list.append(kmeans.inertia_)
mtp.plot(range(1, 11), wcss_list)
mtp.title('Elbow Method - No Scaling')
mtp.xlabel('Number of clusters(k)')
mtp.ylabel('WCSS')
mtp.show()


kmeans1 = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_pred1 = kmeans1.fit_predict(x)


fig = mtp.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
colors = ['blue', 'green', 'red', 'black', 'purple']
for i in range(5):
    ax1.scatter(x[y_pred1 == i, 0], x[y_pred1 == i, 1], x[y_pred1 == i, 2], c=colors[i], label=f'Cluster {i+1}', s=50)
ax1.scatter(kmeans1.cluster_centers_[:, 0], kmeans1.cluster_centers_[:, 1], kmeans1.cluster_centers_[:, 2], c='yellow', s=200, label='Centroid')
ax1.set_title('Without Scaling')
ax1.set_xlabel('Age')
ax1.set_ylabel('Annual Income (k$)')
ax1.set_zlabel('Spending Score')
ax1.legend()


scaler = StandardScaler()
x_scaled = x.copy().astype(float)
x_scaled[:, 1:] = scaler.fit_transform(x[:, 1:])


kmeans2 = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_pred2 = kmeans2.fit_predict(x_scaled)


ax2 = fig.add_subplot(122, projection='3d')
for i in range(5):
    ax2.scatter(x[y_pred2 == i, 0], x[y_pred2 == i, 1], x[y_pred2 == i, 2], c=colors[i], label=f'Cluster {i+1}', s=50)
ax2.set_title('With Scaling (Age unscaled)')
ax2.set_xlabel('Age')
ax2.set_ylabel('Annual Income (k$)')
ax2.set_zlabel('Spending Score')
ax2.legend()


mtp.tight_layout()
mtp.show()


print("Cluster sizes without scaling:", nm.bincount(y_pred1))
print("Cluster sizes with scaling:   ", nm.bincount(y_pred2))
print("\nCluster centers without scaling:\n", kmeans1.cluster_centers_)
print("\nCluster centers with scaling (in original space):")
centers_original = kmeans2.cluster_centers_.copy()
centers_original[:, 1:] = scaler.inverse_transform(kmeans2.cluster_centers_[:, 1:])
print(centers_original)
