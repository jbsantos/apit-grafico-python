import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import pickle


apto = pd.read_csv('/home/jorge/Documentos/python/api/aptoClassificacao/Apto_KNN.csv')
#apto = pd.read_csv('Apto_KNN normalizado.csv')
X = apto.iloc[:, 1:5].values
# print(X)
# kmeans = KMeans(n_clusters=4, init='ndarray[[2,2],[2,8],[8,2],[8,8]]')
kmeans = KMeans(n_clusters=4, init='random')
distancias = kmeans.fit_transform(X)
# print(distancias)
print(legendas)

def teste():


# print ("\n======\n", kmeans.cluster_centers_, "\n======\n")
with  open("kmeans_n_cluster4_padrao.pkl", "wb") as file: pickle.dump(kmeans, file)
legendas = kmeans.labels_
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:,1], s= 50 , c = kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1],  s=300, c= 'red', label = 'Centroids')
plt.title('APTO Clusters Front vs Back')
plt.xlabel('FrontEnd')
plt.ylabel('BackEnd')
plt.legend()
plt.savefig("/home/jorge/Documentos/python/api/static/grafico.jpeg")

plt.show()

#xy = new xy(X[0], X[1])

#return {x=X[:, 0], y=X[:,1], labels=kmeans.labels_}