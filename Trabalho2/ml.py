import numpy
from PIL import Image
from numpy import genfromtxt
from sklearn.neural_network import MLPClassifier

classe = 0
numImages = 1000
nInputs = 3073
numImages = 50000

nTestes = 10000

print("Criando X de treino...")
X = numpy.ones((numImages, 3073))
for num in range(0, numImages):
    im = Image.open('../dataset/train/' + '{:05d}'.format(num) + '.png')
    im = numpy.array(im).flatten()
    X[num, 1:3073] = (im-127)/255


print("Criando Y de treino...")
labelsT = genfromtxt('../dataset/train/labels', delimiter=',')
y = numpy.zeros((numImages))

for num in range(0,numImages):
    if labelsT[num] == classe:
        y[num] = 1

print("Modelando a rede...")
mlp = MLPClassifier(hidden_layer_sizes=(1500),solver='sgd',learning_rate_init=0.01,max_iter=300,verbose=True)
print("Treinando...")
mlp.fit(X, y)
print("Treinado!")

print("Criando X de teste...")
X_test = numpy.ones((nTestes, 3073))
for num in range(0, nTestes):
    im = Image.open('../dataset/test/' + '{:05d}'.format(num) + '.png')
    im = numpy.array(im).flatten()
    X_test[num, 1:3073] = im
X_test = (X_test-127)/255

print("Criando Y de teste...")
labelsTest = genfromtxt('../dataset/test/labels', delimiter=',')
y_test = numpy.zeros((nTestes))

for num in range(0,nTestes):
    if labelsTest[num] == classe:
        y_test[num] = 1

print("Modelando a rede...")
mlp = MLPClassifier(hidden_layer_sizes=(1500),solver='sgd',learning_rate_init=0.01,max_iter=300,verbose=True)
print("Treinando...")
mlp.fit(X, y)
print("Testando...")
print (mlp.score(X_test, y_test))
