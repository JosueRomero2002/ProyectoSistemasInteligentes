from Dataset import MnistDataset
from Network import NeuralNetwork
import numpy as np

mnist_train = MnistDataset()
mnist_train.load("ProyectoNumeros/mnist/dataset/train-images-idx3-ubyte", "ProyectoNumeros/mnist/dataset/train-labels-idx1-ubyte")
mnist_test = MnistDataset()
mnist_test.load("ProyectoNumeros/mnist/dataset/t10k-images-idx3-ubyte", "ProyectoNumeros/mnist/dataset/t10k-labels-idx1-ubyte")


X_train = mnist_train.get_flattened_data()
y_train = mnist_train.get_one_hot_labels()
X_test = mnist_test.get_flattened_data()
y_test = mnist_test.get_one_hot_labels()


input_size = 784 
hidden_size = 64
output_size = 10
learning_rate = 0.1



nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

nn.train(X_train, y_train, epochs=10)


y_test_pred = nn.predict(X_test)
accuracy = np.mean(np.argmax(y_test, axis=1) == y_test_pred)
print("Accuracy: [" +  str(accuracy * 100) + "%]")


