import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Loading and examining the dataset
penguins_df = pd.read_csv("penguins.csv")
penguins_df.head()

penguins_preprocessed = pd.get_dummies(penguins_df)

scaler = StandardScaler()

transformed_Data = scaler.fit_transform(penguins_preprocessed)

inertia = []

for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(transformed_Data)
    
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 10), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

n_clusters = 4

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(transformed_Data)

plt.scatter(penguins_df['culmen_length_mm'], penguins_df['culmen_depth_mm'], c=kmeans.labels_)
plt.xlabel('Culmen Length (mm)')
plt.ylabel('Culmen Depth (mm)')
plt.show()

penguins_df['label'] = kmeans.labels_

stat_penguins = penguins_df.groupby('label').mean(numeric_only=True)

print(stat_penguins)