import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv('../datasets/UnsupervisedLearning/KMeansClustering/Mall_Customers.csv')
data = data[['Annual Income (k$)', 'Spending Score (1-100)']]
data = data.rename(columns={'Annual Income (k$)': 'income', 'Spending Score (1-100)': 'score'})

# plt.scatter(data['income'], data['score'])
# plt.show()


k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
wcss_error = []

# for k in k_values:
#     model = KMeans(n_clusters=k)
#     model.fit(data[['income', 'score']])
#
#     wcss_error.append(model.inertia_)

# plt.plot(k_values, wcss_error)
# plt.xlabel('number of clusters')
# plt.ylabel('wcss error')
# plt.show()


kmeans_model = KMeans(n_clusters=5)
prediction = kmeans_model.fit_predict(data)

data['cluster'] = prediction

cluster1 = data[data['cluster'] == 0]
cluster2 = data[data['cluster'] == 1]
cluster3 = data[data['cluster'] == 2]
cluster4 = data[data['cluster'] == 3]
cluster5 = data[data['cluster'] == 4]

print(cluster1)
print("\n")
print(cluster2)
print("\n")
print(cluster3)
print("\n")
print(cluster4)
print("\n")
print(cluster5)
print("\n")

plt.scatter(cluster1['income'], cluster1['score'])
plt.scatter(cluster2['income'], cluster2['score'])
plt.scatter(cluster3['income'], cluster3['score'])
plt.scatter(cluster4['income'], cluster4['score'])
plt.scatter(cluster5['income'], cluster5['score'])
plt.scatter(kmeans_model.cluster_centers_[:,0], kmeans_model.cluster_centers_[:,1], color = 'Black')
plt.show()
