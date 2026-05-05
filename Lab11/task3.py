import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


np.random.seed(42)
n = 100
data = {
    'student_id': range(1, n+1),
    'GPA': np.round(np.random.uniform(1.5, 4.0, n), 2),
    'study_hours': np.random.randint(2, 40, n),
    'attendance_rate': np.round(np.random.uniform(40, 100, n), 1)
}
df = pd.DataFrame(data)


x = df[['GPA', 'study_hours', 'attendance_rate']].values
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


wcss_list = []
for i in range(2, 7):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x_scaled)
    wcss_list.append(kmeans.inertia_)
mtp.plot(range(2, 7), wcss_list, marker='o')
mtp.title('Elbow Method for Optimal K')
mtp.xlabel('Number of Clusters (K)')
mtp.ylabel('WCSS')
mtp.show()


optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
df['cluster'] = kmeans.fit_predict(x_scaled)


colors = ['blue', 'green', 'red']
for i in range(optimal_k):
    subset = df[df['cluster'] == i]
    mtp.scatter(subset['study_hours'], subset['GPA'], c=colors[i], label=f'Cluster {i+1}', s=80)
mtp.title('Student Clusters Based on Study Hours and GPA')
mtp.xlabel('Study Hours (per week)')
mtp.ylabel('GPA')
mtp.legend()
mtp.show()
