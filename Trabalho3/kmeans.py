# Giovani Nascimento Pereira - 168609
# Carlos Augusto Figueiredo Feire de Carvalho - 165684

import numpy
import sys
from PIL import Image, ImageFilter
from numpy import genfromtxt
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import csv

nTotal = 19924
nTotal = 1000
nFeatures = 2209
nTreino = int(0.9 * nTotal)
nTeste  = 1992

#Atualizar com o caminho do csv no seu diretorio
csvPath = "./dataset/csv.noUp/data.csv"


# Pegando os dados
print('-----------Pegando os dados----------')
reader = csv.reader(open(csvPath, "r"), delimiter=",")
med = list(reader)
data= numpy.array(med).astype("float")

# print(data.shape)
# Dimensao dos dados completos: (19924, 2209) -> as expected
# 19924 documentos
# 2209 features

# Separa em teste
print('-----------Montando Teste----------')
X_teste = numpy.ones((nTeste, nFeatures + 1))
for num in range(0, nTeste):
	X_teste[num, 1:(nFeatures+1)] = data[num, 0:nFeatures]

# Separa em treino
print('-----------Montando Treino----------')
X_treino = numpy.ones((nTreino, nFeatures + 1))
for num in range(0, nTreino):
	X_treino[num, 1:(nFeatures+1)] = data[num + nTeste, 0:nFeatures]


# Define modelo do KMeans - parâmetros:
# KMeans(n_clusters=8,
#		init=’k-means++’,
#		n_init=10,
#		max_iter=300,
#		tol=0.0001,
#		precompute_distances=’auto’,
#		verbose=0, random_state=None,
#		copy_x=True, n_jobs=1, algorithm=’auto’)

kmeans = KMeans(random_state = 1)

#  Primeira parte do treinamento, treinamos usando kMeans e incrementando o
# numero de clusters a cada iteração
#

maior = 0
id_maior = 2

for i in range (2, 3):
	# altera o numero de clusters
	kmeans = KMeans(n_clusters=i)
	kmeans.fit(X_treino)
	# predição dos labels para o dataset de teste
	label_teste = kmeans.predict(X_teste)
	print('Numero de clusters: ', i)
	# calcula e imprime os silhouette_score desse modelo com esse numero de clusters
	silhouette_score_new = silhouette_score(X_teste, label_teste)
	print(silhouette_score_new, '\n')
	# confere qual o numero de clusters que deu melhor resultado
	if(silhouette_score_new > maior):
		maior = silhouette_score_new
		id_maior = i

print('Melhor cluster: ', id_maior, '\t com silhouette_score de :', maior)

# Fim do primeiro teste
#
#

#  Segundo teste, pegamos o melhor resultado do primeiro cluster e mudamos
# o init, que por padrão era kMeans++, p/ random
#

kmeans2 = KMeans(random_state = 1, init = 'random', n_clusters = id_maior)
kmeans2.fit(X_treino)
label_teste = kmeans2.fit(X_treino)
print('Mudando o método de inicialização do centroides para random: ')
silhouette_score_new = silhouette_score(X_teste, label_teste)
print(silhouette_score_new, '\n')

# Fim do segundo teste
#
#

# Fazer um teste com mini-batch k-means
#
#

miniKmeans = MiniBatchKMeans(n_clusters = 100)
miniKmeans.fit(X_treino)
label_mini = miniKmeans.predict(X_teste)
print('Mudando para MiniBatchKMeans temos: ', silhouette_score(X_teste, label_mini), '\n')

# Fazer um teste com Aglomerative
# http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
#
#

agglomerativeClustering = AgglomerativeClustering(n_clusters = 100)
agglomerativeClustering.fit(X_treino)
label_agglomerative = AgglomerativeClustering.predict(X_teste)
print('Mudando para AgglomerativeClustering temos: ', silhouette_score(X_teste, label_agglomerative), '\n')

#
#
#
#
