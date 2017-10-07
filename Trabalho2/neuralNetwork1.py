# Giovani Nascimento Pereira - 168609
# Carlos Augusto Figueiredo Feire de Carvalho - 

import numpy
import sys
from PIL import Image, ImageFilter
from numpy import genfromtxt
from sklearn.neural_network import MLPClassifier

classe = 0			# O que eu to tentando reconhecer
numImages = 1000	# Número de imagens para treinar a rede
nInputs = 3073		# dimensao da imagem
nTestes = 1000		# Numero de imagens de teste


# Definindo o conjunto de testes
if (len(sys.argv) > 1):
	if (sys.argv[1] == 'small'):
		addr = './dataset.small'
		print("Rodando com o Small Dataset - número de imagens = 1000")
		numImages = 1000
		nTestes = 1000

	elif (sys.argv[1] == 'full'):
		addr = './dataset.noUp'
		print("WOOHOOOO RODANDO COM TODOS OS DADOOOS")
		numImages = 50000
		nTestes = 10000

	else:
		addr = './dataset.noUp'
		print("Rodando com o Dataset completo")
		if (int(sys.argv[1]) > 0):
			numImages = int(sys.argv[1])
			print("Usando " + str(numImages) + " imagens")
			if (len(sys.argv) > 2):
				nTestes = int(sys.argv[2])
			else:
				nTestes = 5000
		else:
			print("Usando número default de imagens = 10000")
			numImages = 10000
			nTestes = 1000

else:
	print("Rodando no mode default - Dataset Completo com 1000 imagens")
	addr = './dataset.noUp'
	numImages = 1000
	nTestes = 1000

print("\n---------------------------")
print("Numero de Imagens de Treino: " + str(numImages))
print("Numero de Imagens de Teste: "  + str(nTestes))
print("Dataset: " + addr)
print("---------------------------\n")

print("Criando X de treino...")
X = numpy.ones((numImages, 3073))
for num in range(0, numImages):
    im = Image.open(addr + '/train/' + '{:05d}'.format(num) + '.png')
    #Applying filters
    im = im.filter(ImageFilter.DETAIL)
    im = numpy.array(im).flatten()
    X[num, 1:3073] = (im-127)/255


print("Criando Y de treino...")
labelsT = genfromtxt(addr + '/train/labels', delimiter=',')
y = numpy.zeros((numImages))

for num in range(0,numImages):
    if labelsT[num] == classe:
        y[num] = 1

print("Modelando a rede...")
mlp = MLPClassifier(hidden_layer_sizes=(1500),solver='sgd',learning_rate_init=0.01,max_iter=300,verbose=True)
mlp.fit(X, y)	#Treinando a Rede

print("Criando X de teste...")
X_test = numpy.ones((nTestes, 3073))
for num in range(0, nTestes):
    im = Image.open(addr + '/test/' + '{:05d}'.format(num) + '.png')
    #Applying filters
    im = im.filter(ImageFilter.DETAIL)
    im = numpy.array(im).flatten()
    X_test[num, 1:3073] = im
X_test = (X_test-127)/255

print("Criando Y de teste...")
labelsTest = genfromtxt(addr + '/test/labels', delimiter=',')
y_test = numpy.zeros((nTestes))

for num in range(0,nTestes):
    if labelsTest[num] == classe:
        y_test[num] = 1

print (mlp.score(X_test, y_test))
