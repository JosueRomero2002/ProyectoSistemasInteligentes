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

        self.capa1.weights_loader("ProyectoNumeros/savedweights")
        self.capa2.weights_loader("ProyectoNumeros/savedweights")
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

    def train(self, X, y, epochs, batch_size=64):
        num_samples = X.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)  # Shuffle data
            for i in range(0, num_samples, batch_size):
                batch_X = X[indices[i:i+batch_size]]
                batch_y = y[indices[i:i+batch_size]]

                y_pred = self.forward(batch_X)
                loss = self.loss_function.compute_loss(batch_y, y_pred)
                self.backward(batch_X, batch_y, y_pred)

            if epoch % 10 == 0:  
                print(f"Epoch [{epoch}] ---- Loss: [{loss:.4f}]")
                self.capa1.weights_saver("ProyectoNumeros/savedweights")
                self.capa2.weights_saver("ProyectoNumeros/savedweights")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)



    # def train(self, X, y, epochs):
        
    #     for epoch in range(epochs):
    #         y_pred = self.forward(X)
    #         loss = self.loss_function.compute_loss(y, y_pred)
    #         self.backward(X, y, y_pred)
           
            
    #         print(f"Epoch [{epoch}] ---- Loss: [{loss:.4f}]")
    #         self.capa1.weights_saver("ProyectoNumeros/savedweights")
    #         self.capa2.weights_saver("ProyectoNumeros/savedweights")
                
                