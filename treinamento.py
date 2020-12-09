from sklearn.cluster import KMeans




def objGrafico():
    import numpy as np
    import json

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    url = "https://run.mocky.io/v3/043a3b23-4e31-4000-9073-dec4d6f97054"
    infoNotas = salvaDados()
    a = np.array(kmeansLabel())
    img_json = np.array(jsonImagem())
    json_dump = json.dumps({'legenda': a, 'imagem': img_json, 'infoNotas': infoNotas, 'url': url}, cls=NumpyEncoder)

    return json_dump



def salvaDados():

    import numpy as np
    import requests
    import pandas as pd
    import json
    import requests
    import re

    #r = requests.get("https://run.mocky.io/v3/4e722972-ea21-466c-860d-4e08c3f5c4eb").json()
    r = requests.get("https://run.mocky.io/v3/043a3b23-4e31-4000-9073-dec4d6f97054").json()
    data = np.array(r)
    dataFrame = pd.DataFrame(data, columns = ['id', 'backend', 'frontEnd', 'cv', 'Entrevista'])


    #print(dataFrame)
    json = dataFrame.to_json()
    #print(json)
    json_split = dataFrame.to_json(orient ='split')
    #print("json_split = ", json_split, "\n")

    json_records = dataFrame.to_json(orient ='records')
    #print("json_records = ", json_records, "\n")

    json_index = dataFrame.to_json(orient ='index')
    #print("json_index = ", json_index, "\n")

    json_columns = dataFrame.to_json(orient ='columns')
    #print("json_columns = ", json_columns, "\n")

    #print("json_values = ", json_values, "\n")

    json_table = dataFrame.to_json(orient ='table')
    #print("json_table = ", json_table, "\n")

    dados = pd.DataFrame(dataFrame)
    df = pd.DataFrame(dataFrame)
    df.to_csv(index=False)
    df.to_csv('aptoClassificacao/Apto_KNN.csv', index=False, encoding='utf-8')
    return r
    #apto = pd.read_csv('aptoClassificacao/Apto_KNN.csv')
    #json_values = apto.to_json(orient ='values')
    #print(json_values)

def kmeansLabel():

    import numpy as np
    import pickle
    import pandas as pd
    #apto = pd.read_csv('/home/jorge/Documentos/python/api/aptoClassificacao/Apto_KNN.csv')
    apto = pd.read_csv('aptoClassificacao/Apto_KNN.csv')
    #print(apto)
    X = apto.iloc[:, 1:5].values
    # print(X)
    # kmeans = KMeans(n_clusters=4, init='ndarray[[2,2],[2,8],[8,2],[8,8]]')
    kmeans = KMeans(n_clusters=4, init='random')
    distancias = kmeans.fit_transform(X)
    #print(distancias)
    legendas = kmeans.labels_
    #print(legendas.shape)


    ts = legendas.tostring()
    legenda_texto = np.fromstring(ts, dtype=int)
    #print(legenda_texto.to_json())
    # print ("\n======\n", kmeans.cluster_centers_, "\n======\n")
    with  open("kmeans_n_cluster4_padrao.pkl", "wb") as file: pickle.dump(kmeans, file)

    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:,1], s= 100 , c = kmeans.labels_)
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1],  s=300, c= 'red', label = 'Centroids')
    plt.title('APTO Clusters Front vs Back')
    plt.xlabel('FrontEnd')
    plt.ylabel('BackEnd')
    plt.legend()
    plt.savefig("static/grafico.jpeg")
    plt.close()
    return legendas


def jsonImagem():

    import base64
    data = {}

    with open('static/grafico.jpeg', mode='rb') as file:
        img = file.read()
    data['img'] = base64.encodebytes(img).decode('utf-8').replace('\n', '')

    return data
