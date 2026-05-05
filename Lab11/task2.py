import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder


data = {
    'vehicle_serial_no': [5, 3, 8, 2, 4, 7, 6, 10, 1, 9],
    'mileage': [150000, 120000, 250000, 80000, 100000, 220000, 180000, 300000, 75000, 280000],
    'fuel_efficiency': [15, 18, 10, 22, 20, 12, 16, 8, 24, 9],
    'maintenance_cost': [5000, 4000, 7000, 2000, 3000, 6500, 5500, 8000, 1500, 7500],
    'vehicle_type': ['SUV', 'Sedan', 'Truck', 'Hatchback', 'Sedan', 'Truck', 'SUV', 'Truck', 'Hatchback', 'SUV']
}


df = pd.DataFrame(data)


le = LabelEncoder()
df['vehicle_type_encoded'] = le.fit_transform(df['vehicle_type'])


x = df[['mileage', 'fuel_efficiency', 'maintenance_cost', 'vehicle_type_encoded']].values


wcss_list = []
for i in range(1, 6):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss_list.append(kmeans.inertia_)
mtp.plot(range(1, 6), wcss_list)
mtp.title('Elbow Method - No Scaling')
mtp.xlabel('Number of clusters(k)')
mtp.ylabel('WCSS')
mtp.show()


kmeans1 = KMeans(n_clusters=3, init='k-means++', random_state=42)
y_pred1 = kmeans1.fit_predict(x)


scaler = StandardScaler()
x_scaled = x.copy().astype(float)
x_scaled[:, :3] = scaler.fit_transform(x[:, :3])


kmeans2 = KMeans(n_clusters=3, init='k-means++', random_state=42)
y_pred2 = kmeans2.fit_predict(x_scaled)


fig, axes = mtp.subplots(1, 2, figsize=(14, 5))
colors = ['blue', 'green', 'red']


for i in range(3):
    axes[0].scatter(x[y_pred1 == i, 0], x[y_pred1 == i, 1], c=colors[i], label=f'Cluster {i+1}', s=100)
axes[0].scatter(kmeans1.cluster_centers_[:, 0], kmeans1.cluster_centers_[:, 1], c='yellow', s=300, marker='*', label='Centroid')
axes[0].set_title('Without Scaling')
axes[0].set_xlabel('Mileage')
axes[0].set_ylabel('Fuel Efficiency')
axes[0].legend()


for i in range(3):
    axes[1].scatter(x[y_pred2 == i, 0], x[y_pred2 == i, 1], c=colors[i], label=f'Cluster {i+1}', s=100)
axes[1].set_title('With Scaling (vehicle_type unscaled)')
axes[1].set_xlabel('Mileage')
axes[1].set_ylabel('Fuel Efficiency')
axes[1].legend()


mtp.tight_layout()
mtp.show()


print("Cluster labels without scaling:", y_pred1)
print("Cluster labels with scaling:   ", y_pred2)
print("\nCluster sizes without scaling:", np.bincount(y_pred1))
print("Cluster sizes with scaling:   ", np.bincount(y_pred2))


df['cluster_no_scaling'] = y_pred1
df['cluster_with_scaling'] = y_pred2
print("\nVehicle assignments:")
print(df[['vehicle_serial_no', 'vehicle_type', 'mileage', 'fuel_efficiency', 'maintenance_cost', 'cluster_no_scaling', 'cluster_with_scaling']])