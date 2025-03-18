import numpy as np
from DenseLayer import DenseLayer
from Activation import ReLUActivation, SoftmaxActivation
from Dataset import MnistDataset


mnist_train = MnistDataset()
mnist_train.load("ProyectoNumeros/mnist/dataset/train-images-idx3-ubyte", "ProyectoNumeros/mnist/dataset/train-labels-idx1-ubyte")
mnist_test = MnistDataset()
mnist_test.load("ProyectoNumeros/mnist/dataset/t10k-images-idx3-ubyte", "ProyectoNumeros/mnist/dataset/t10k-labels-idx1-ubyte")


X_train = mnist_train.get_flattened_data()
y_train = mnist_train.get_one_hot_labels()
X_test = mnist_test.get_flattened_data()
y_test = mnist_test.get_one_hot_labels()



# Ajustes de Red
inputDataLength = 784
hiddenSize = 128
outputSize = 10
learningRate = 0.1


# Capa 1
capa1 = DenseLayer(inputDataLength, hiddenSize)
salida_capa1 = capa1.forward(X_train[:10]) 

relu1 = ReLUActivation()
salida_relu1 = relu1.forward(salida_capa1)  

print("Salida de la capa 1 -------------------")
# print(salida_relu1)

# Capa 2
capa2 = DenseLayer(hiddenSize, hiddenSize)
salida_capa2 = capa2.forward(salida_relu1)  

relu2 = ReLUActivation()
salida_relu2 = relu2.forward(salida_capa2) 

print("Salida de la capa 2 -------------------")
# print(salida_relu2)

# Capa de salida
capaSalida = DenseLayer(hiddenSize, outputSize)
salida_final = capaSalida.forward(salida_relu2) 

softmax = SoftmaxActivation()
salida_softmax = softmax.forward(salida_final)  

print("Salida de la capa de salida -------------------")
# print(salida_final)

print("Salida de la capa de salida con softmax -------------------")
print(salida_softmax)




