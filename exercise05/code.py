from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

df = pd.read_csv("Avm_Musterileri.csv")
df.head()

plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

df.rename(columns={'Annual Income (k$)': 'Income'}, inplace=True)
df.rename(columns={'Spending Score (1-100)': 'Score'}, inplace=True)

scaler = MinMaxScaler()

scaler.fit(df[['Income']])
df['Income'] = scaler.transform(df[['Income']])

scaler.fit(df[['Score']])
df['Score'] = scaler.transform(df[['Score']])

k_range = range(1,11)

list_dict = []

for k in k_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df[['Income', 'Score']])
    list_dict.append(kmeans.inertia_)

plt.xlabel('K')
plt.ylabel('Distortion değeri (Inertia)')
plt.plot(k_range, list_dict)
plt.show()

kmeans = KMeans(n_clusters=5)
y_predicted = kmeans.fit_predict(df[['Income', 'Score']])
y_predicted

df['Cluster'] = y_predicted
df.head()

kmeans.cluster_centers_

df1 = df[df.Cluster==0]
df2 = df[df.Cluster==1]
df3 = df[df.Cluster==2]
df4 = df[df.Cluster==3]
df5 = df[df.Cluster==4]

plt.xlabel("Income")
plt.ylabel("Score")

plt.scatter(df1['Income'], df1['Score'], color='green')
plt.scatter(df2['Income'], df2['Score'], color='red')
plt.scatter(df3['Income'], df3['Score'], color='blue')
plt.scatter(df4['Income'], df4['Score'], color='black')
plt.scatter(df5['Income'], df5['Score'], color='purple')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color='yellow',
            marker = 'X', label = 'centroid')
plt.legend()
plt.show()
