# Giovani Nascimento Pereira - 168609
# Carlos Augusto Figueiredo Feire de Carvalho - 


import numpy
import sys
from PIL import Image, ImageFilter
from numpy import genfromtxt
from sklearn.cluster import KMeans
import csv

nTotal = 19924
nFeatures = 2209
nTreino = int(0.9 * nTotal)
nTeste  = int(nTotal - nTreino)


# Pegando os dados
print('-----------Pegando os dados----------')
reader = csv.reader(open("./dataset/csv.noUp/data.csv", "r"), delimiter=",")
med = list(reader)
data= numpy.array(med).astype("float")

# print(data.shape)
# Dimensao dos dados completos: (19924, 2209) -> as expected
# 19924 documentos
# 2209 features

# Separa em treino
print('-----------Montando Treino----------')
X_treino = numpy.ones((nTreino, nFeatures + 1))
for num in range(0, nTreino):
	X_treino[num, 1:(nFeatures+1)] = data[num, 0:nFeatures]

#Separa em teste
print('-----------Montando Teste----------')
X_teste = numpy.ones((nTeste, nFeatures + 1))
for num in range(0, nTeste):
	X_teste[num, 1:(nFeatures+1)] = data[num + nTreino, 0:nFeatures]


# Define modelo do KMeans - parâmetros:
# KMeans(n_clusters=8, 
#		init=’k-means++’, 
#		n_init=10, 
#		max_iter=300, 
#		tol=0.0001, 
#		precompute_distances=’auto’, 
#		verbose=0, random_state=None, 
#		copy_x=True, n_jobs=1, algorithm=’auto’)
kmeans = KMeans(n_clusters=5, random_state=0, verbose=True)
kmeans.fit(X_treino)

#kmeans.fit(X_treino)
kmeans.predict(X_teste)

print('Treino: ', kmeans.score(X_treino), '\tTeste: ', kmeans.score(X_teste))

