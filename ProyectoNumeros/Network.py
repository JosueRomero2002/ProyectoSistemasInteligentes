import numpy as np
from DenseLayer import DenseLayer
from Activation import ReLUActivation, SoftmaxActivation
from LossFunctions import CrossEntropyLoss

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.capa1 = DenseLayer(input_size, hidden_size)
        self.activation1 = ReLUActivation()
        self.capa2 = DenseLayer(hidden_size, output_size)
        self.activation2 = SoftmaxActivation()
        self.loss_function = CrossEntropyLoss()
        self.learning_rate = learning_rate

        #Weights loaded
        # self.capa1.load_weights("ProyectoNumeros/savedweights")


    def forward(self, X):
        self.z1 = self.capa1.forward(X)
        self.a1 = self.activation1.forward(self.z1)
        self.z2 = self.capa2.forward(self.a1)
        self.a2 = self.activation2.forward(self.z2)
        return self.a2

    def backward(self, X, y_true, y_pred):
        grad_loss = self.loss_function.compute_gradient(y_true, y_pred)
        grad_a2 = self.activation2.backward(grad_loss, y_pred)
        grad_z2 = self.capa2.backward(grad_a2, self.learning_rate)
        grad_a1 = self.activation1.backward(grad_z2)
        self.capa1.backward(grad_a1, self.learning_rate)

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.loss_function.compute_loss(y, y_pred)
            self.backward(X, y, y_pred)
            if epoch % 100 == 0:
                print(f"Epoch [{epoch}] ---- Loss: [{loss:.4f}]")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)
