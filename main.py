import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Carregando o dataset
df = pd.read_csv(r'segmentation data.csv')

# Removendo a coluna 'ID'
df.drop('ID', axis=1, inplace=True)

# Treinando o modelo K-Means
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
y_predict = kmeans.fit_predict(df)

# Calculando o coeficiente de silhueta
silhouette_avg = silhouette_score(df, y_predict)
print(f'Coeficiente de Silhueta: {silhouette_avg}')

# Visualizando os clusters
plt.scatter(df['Age'][y_predict == 0], df['Income'][y_predict == 0], s=100, c='blue', label='Cluster 1')  # 1º cluster
plt.scatter(df['Age'][y_predict == 1], df['Income'][y_predict == 1], s=100, c='green', label='Cluster 2')  # 2º cluster
plt.scatter(df['Age'][y_predict == 2], df['Income'][y_predict == 2], s=100, c='red', label='Cluster 3')  # 3º cluster
plt.scatter(df['Age'][y_predict == 3], df['Income'][y_predict == 3], s=100, c='cyan', label='Cluster 4')  # 4º cluster
plt.scatter(df['Age'][y_predict == 4], df['Income'][y_predict == 4], s=100, c='magenta', label='Cluster 5')  # 5º cluster

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroid')

plt.title('Clusters de Clientes')
plt.xlabel('Idade dos Clientes')
plt.ylabel('Renda dos Clientes')
plt.legend()
plt.show()
