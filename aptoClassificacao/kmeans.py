import requests
import pickle
with  open("kmeans_n_cluster4_padrao.pkl", "rb") as arquivo:
 kmeans_modelo= pickle.load(arquivo)

r = requests.get("https://run.mocky.io/v3/9e77d373-7e13-4276-8dc5-8d2b66f120a8")
print(r.status_code)
print(r.encoding)
print(r.apparent_encoding)
print(r.text)

data = r.json() 
print(data)
[
          [4.1, 8.1, 5, 2],
          [2.2, 5.3, 5, 3],
          [7.4, 4.4, 2.4, 2.2],
          [3.2, 5.2, 5.4, 6.2],
          [4.5, 6.2, 3.4, 2.5],
          [6.4, 5.4, 3.4, 3.4],
          [9.4, 9.6, 4.4, 2.7],
          [1, 1, 1, 1],
          [10, 10, 5, 3]
]

previsto = kmeans_modelo.predict(r.json())
print(previsto)