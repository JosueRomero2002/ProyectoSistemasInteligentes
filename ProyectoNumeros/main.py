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
hidden_size = 128
output_size = 10
learning_rate = 0.01
epochs=20
batchSize = 64


saveandprinteach = 10 #not yet implemented


# LossRegulation = 1 or 2 (using l1 or l2) (l1 is optional but l2 is obligatory)



nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

nn.train(X_train, y_train, epochs, batchSize, y_test, X_test)


y_test_pred = nn.predict(X_test)
accuracy = np.mean(np.argmax(y_test, axis=1) == y_test_pred)
print("Accuracy: [" +  str(accuracy * 100) + "%]")

# Use Mathlib to make the graphs
