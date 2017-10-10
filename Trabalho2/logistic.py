# Giovani Nascimento Pereira - 168609
# Carlos Augusto Figueiredo Feire de Carvalho - 

import numpy
import sys
from PIL import Image, ImageFilter
from numpy import genfromtxt
from sklearn import linear_model, datasets

classe = 0			# O que eu to tentando reconhecer
numImages = 1000	# Numero de imagens para treinar a rede
nInputs = 3073		# dimensao da imagem
nTestes = 1000		# Numero de imagens de teste


# Definindo o conjunto de testes
if (len(sys.argv) > 1):
	if (sys.argv[1] == 'small'):
		addr = './dataset.noUp'
		print("Rodando com o Small Dataset - numero de imagens = 1000")
		numImages = 1000
		nTestes = 10000

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
			print("Usando numero default de imagens = 10000")
			numImages = 10000
			nTestes = 1000

else:
	print("Rodando no mode default - Dataset Completo com 1000 imagens")
	addr = './dataset.noUp'
	numImages = 1000
	nTestes = 1000

print("\n---------------------------\n")
print("Numero de Imagens de Treino: " + str(numImages))
print("Numero de Imagens de Teste: "  + str(nTestes))
print("Dataset: " + addr)
print("\n---------------------------\n")

print("Criando X de treino...")
X = numpy.ones((numImages, 3072))
for num in range(0, numImages):
    im = Image.open(addr + '/train/' + '{:05d}'.format(num) + '.png')
    im = im.filter(ImageFilter.EMBOSS)
    im = numpy.array(im).flatten()
    X[num] = (im - 127) / 255


print("Criando Y de treino...")
y_in = genfromtxt(addr + '/train/labels', delimiter=',')
y = y_in[0:numImages] 


print("Modelando a rede...")

logreg = linear_model.LogisticRegression(solver = 'sag', multi_class = 'multinomial',C = 10e-5, verbose = 1, tol = 1e-20, max_iter=30)
print("Treinando...")
logreg.fit(X, y)
print("Treinado!")

print("Criando X de teste...")
X_test = numpy.ones((nTestes, 3072))
for num in range(0, nTestes):
    im = Image.open(addr + '/test/' + '{:05d}'.format(num) + '.png')
    im = im.filter(ImageFilter.EMBOSS)
    im = numpy.array(im).flatten()
    X_test[num] = (im - 127) / 255

print("Criando Y de teste...")
y_test_in = genfromtxt(addr + '/test/labels', delimiter=',')
y_test = y_test_in[0:nTestes]
y_predict = logreg.predict(X_test)
numpy.savetxt('output1.txt', y_predict)
numpy.savetxt('output2.txt', y_test)
print("Testando")
print(logreg.score(X_test, y_test))
print(logreg.score(X, y))

