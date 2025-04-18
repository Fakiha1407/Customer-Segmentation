#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
#code
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Synthetic Customer Data
data = {
    'CustomerID': range(1, 201),
    'Age': pd.Series([20 + i % 30 for i in range(200)]),
    'Annual Income (k$)': pd.Series([15 + (i * 3) % 60 for i in range(200)]),
    'Spending Score (1-100)': pd.Series([5 + (i * 7) % 95 for i in range(200)])
}

df = pd.DataFrame(data)

# Exploratory Data Analysis
print(df.head())
sns.pairplot(df.drop('CustomerID', axis=1))
plt.suptitle("Pairplot of Customer Features", y=1.02)
plt.show()

#  Feature Scaling
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Finding Optimal K using Elbow Method
inertia = []
k_range = range(1, 11)
for i in k_range:
    km = KMeans(n_clusters=i,n_init=10, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

#Apply KMeans with Optimal K
k_optimal = 4
kmeans = KMeans(n_clusters=k_optimal,n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

#Visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='Annual Income (k$)', 
    y='Spending Score (1-100)', 
    hue='Cluster',
    data=df, 
    palette='Set2', 
    s=100
)
plt.title('Customer Segments')
plt.legend(title='Cluster')
plt.show()

# Save clustered data 
df.to_csv("clustered_customers.csv", index=False)


# In[ ]:





# In[ ]:




