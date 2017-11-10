# Giovani Nascimento Pereira - 168609
# Carlos Augusto Figueiredo Feire de Carvalho - 165684

import numpy
import sys
from PIL import Image, ImageFilter
from numpy import genfromtxt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA
import timeit
import csv
import kmedoids
import numpy
import sys
from PIL import Image, ImageFilter
from numpy import genfromtxt
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_samples, silhouette_score
import csv

from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.cm as cm


nTotal = 19924
nTotal = 1000
nFeatures = 2209
nTreino = nTotal

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

# Separa em treino
print('-----------Montando Treino----------')
X_treino = numpy.ones((nTreino, nFeatures + 1))
for num in range(0, nTreino):
	X_treino[num, 1:(nFeatures+1)] = data[num, 0:nFeatures]



print(X_treino.shape)
for i in range (1, 100):
    pca = PCA(n_components= i/100, svd_solver='full')
    X_treino_pca = pca.fit_transform(X_treino)
    kmeans_pca = KMeans(n_clusters=78)
    label_teste = kmeans_pca.fit_predict(X_treino_pca)

    #print('Aplicando PCA no nosso melhor modelo com clusters:', 78, 'e variancia: ', i/10)
    # calcula e imprime os silhouette_score desse modelo com esse numero de clusters
    silhouette_score_new = silhouette_score(X_treino, label_teste)
    print(X_treino_pca.shape[1], silhouette_score_new)






# Define modelo do KMeans - parâmetros:
# KMeans(n_clusters=8,
#		init=’k-means++’,
#		n_init=10,
#		max_iter=300,
#		tol=0.0001,
#		precompute_distances=’auto’,
#		verbose=0, random_state=None,
#		copy_x=True, n_jobs=1, algorithm=’auto’)

kmeans = KMeans()

#  Primeira parte do treinamento, treinamos usando kMeans e incrementando o
# numero de clusters a cada iteração
#

# Usando o metodo do elbow
# Reference1: https://pythonprogramminglanguage.com/kmeans-elbow-method/
# Reference2: http://www.awesomestats.in/python-cluster-validation/

maior = 0
id_maior = 2

for i in range (100, 101):
	# altera o numero de clusters
	kmeans = KMeans(n_clusters=i)
	label = kmeans.fit_predict(X_treino)
	print('Numero de clusters: ', i)
	# calcula e imprime os silhouette_score desse modelo com esse numero de clusters
	silhouette_score_new = silhouette_score(X_treino, label)
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
#timeit.timeit('kmeans2.fit(X_treino)')
label = kmeans2.fit_predict(X_treino)
print('Mudando o método de inicialização do centroides para random: ')
silhouette_score_new = silhouette_score(X_treino, label)
print(silhouette_score_new, '\n')

# Fim do segundo teste
#
#

# Fazer um teste com mini-batch k-means
#
#

miniKmeans = MiniBatchKMeans(n_clusters = 100)
label_mini = miniKmeans.fit_predict(X_treino)
print('Mudando para MiniBatchKMeans temos: ', silhouette_score(X_treino, label_mini), '\n')

# MEDOIDS
# https://github.com/salspaugh/machine_learning/blob/master/clustering/kmedoids.py

distancia_treino = pairwise_distances(X_treino, metric='euclidean')
clusters, medoids = kmedoids.cluster(distancia_treino, 100)
print('Numero de clusters: ', i)
print('Mudando para K-Medoids temos: ', silhouette_score(distancia_treino, clusters), '\n')

#	APLICANDO PCA
#
#



print(X_treino.shape)
for i in range (1, 11):
    pca = PCA(n_components= i/10, svd_solver='full')
    X_treino_pca = pca.fit_transform(X_treino)
    print('shape dos dados: ', X_treino_pca.shape)
    for j in range(1, 20):
        kmeans_pca = KMeans(n_clusters=10 * (j - 1) + 2)
        label_teste = kmeans_pca.fit_predict(X_treino_pca)

        print('Aplicando PCA no nosso melhor modelo com clusters:', 10*(j - 1) + 2, 'e variancia: ', i/10)
        # calcula e imprime os silhouette_score desse modelo com esse numero de clusters
        silhouette_score_new = silhouette_score(X_treino_pca, label_teste)
        print(silhouette_score_new, '\n')





#		teste mais detalhado no range que vimos ter melhor resultado
#
#

for i in range (1, 21):
    pca = PCA(n_components=  i/100, svd_solver='full')
    X_treino_pca = pca.fit_transform(X_treino)
    print('shape dos dados: ', X_treino_pca.shape)
    for j in range(1, 30):
        kmeans_pca = KMeans(n_clusters= (j - 1) + 2)
        label_teste = kmeans_pca.fit_predict(X_treino_pca)

        print('Aplicando PCA no nosso melhor modelo com clusters:', (j - 1) + 2, 'e variancia: ', i/100)
        # calcula e imprime os silhouette_score desse modelo com esse numero de clusters
        silhouette_score_new = silhouette_score(X_treino_pca, label_teste)
        print(silhouette_score_new, '\n')

#
#
