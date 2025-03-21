from Dataset import MnistDataset
from Network import NeuralNetwork
import numpy as np
from Optimizer_Adam import Optimizer_Adam

mnist_train = MnistDataset()
mnist_train.load("ProyectoNumeros/Mnist/dataset/train-images-idx3-ubyte", "ProyectoNumeros/Mnist/dataset/train-labels-idx1-ubyte")
mnist_test = MnistDataset()
mnist_test.load("ProyectoNumeros/Mnist/dataset/t10k-images-idx3-ubyte", "ProyectoNumeros/Mnist/dataset/t10k-labels-idx1-ubyte")

X_train = mnist_train.get_flattened_data()
y_train = mnist_train.get_one_hot_labels()
X_test = mnist_test.get_flattened_data()
y_test = mnist_test.get_one_hot_labels()

input_size = 784
hidden_size = 128
output_size = 10
learning_rate = 0.001
epochs = 100
batchSize = 64



saveandprinteach = 10 #not yet implemented



# LossRegulation = 1 or 2 (using l1 or l2) (l1 is optional but l2 is obligatory)



optimizer = Optimizer_Adam(learning_rate=learning_rate, decay=1e-3)
nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate, optimizer)

nn.train(X_train, y_train, epochs, batchSize, y_test, X_test)

y_test_pred = nn.predict(X_test)
accuracy = np.mean(np.argmax(y_test, axis=1) == y_test_pred)
print("Accuracy: [" + str(accuracy * 100) + "%]")



# Use Mathlib to make the graphs
