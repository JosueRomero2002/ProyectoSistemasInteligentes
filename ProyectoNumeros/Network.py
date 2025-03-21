import numpy as np
from DenseLayer import DenseLayer
from Activation import ReLUActivation, SoftmaxActivation
from LossFunctions import CrossEntropyLoss

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, optimizer=None):
        self.capa1 = DenseLayer(input_size, hidden_size)
        self.capa1.weights_loader("ProyectoNumeros/savedweights_capa1")
        self.activation1 = ReLUActivation()
        self.capa2 = DenseLayer(hidden_size, output_size)
        self.capa2.weights_loader("ProyectoNumeros/savedweights_capa2")
        self.activation2 = SoftmaxActivation()
        self.loss_function = CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.optimizer = optimizer

    def forward(self, X):
        self.z1 = self.capa1.forward(X)
        self.a1 = self.activation1.forward(self.z1)
        self.z2 = self.capa2.forward(self.a1)
        self.a2 = self.activation2.forward(self.z2)
        return self.a2

    def backward(self, X, y_true, y_pred, lambda_l2=0.0001):
        grad_loss = self.loss_function.compute_gradient(y_true, y_pred)
        grad_a2 = self.activation2.backward(grad_loss, y_pred)
        grad_z2 = self.capa2.backward(grad_a2, lambda_l2)  
       
        # self.capa2.weights -= lambda_l2 * self.capa2.weights
        # self.capa1.weights -= lambda_l2 * self.capa1.weights
        
        grad_a1 = self.activation1.backward(grad_z2)
        self.capa1.backward(grad_a1, lambda_l2) 
    # def backward(self, X, y_true, y_pred, lambda_l2=0.0001):
    #     grad_loss = self.loss_function.compute_gradient(y_true, y_pred)
    #     grad_a2 = self.activation2.backward(grad_loss, y_pred)
    #     grad_z2 = self.capa2.backward(grad_a2)

    #     self.capa2.weights -= lambda_l2 * self.capa2.weights
    #     self.capa1.weights -= lambda_l2 * self.capa1.weights

    #     grad_a1 = self.activation1.backward(grad_z2)
    #     self.capa1.backward(grad_a1)

    def train(self, X, y, epochs, batch_size, ytest, X_test, saveandprinteach):
        num_samples = X.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            for i in range(0, num_samples, batch_size):
                batch_X = X[indices[i:i+batch_size]]
                batch_y = y[indices[i:i+batch_size]]

                y_pred = self.forward(batch_X)
                loss = self.loss_function.compute_loss(batch_y, y_pred)
                self.backward(batch_X, batch_y, y_pred)

                if self.optimizer is not None:
                    self.optimizer.pre_update_params()
                    self.optimizer.update_params(self.capa1)
                    self.optimizer.update_params(self.capa2)
                    self.optimizer.post_update_params()

            if epoch % saveandprinteach == 0:
                y_test_pred = self.predict(X_test)
                accuracy = np.mean(np.argmax(ytest, axis=1) == y_test_pred)
                print(f"Epoch [{epoch}] ---- Loss: [{loss:.4f}] --Accuracy: [{accuracy*100}%]")
                self.capa1.weights_saver("ProyectoNumeros/savedweights_capa1")
                self.capa2.weights_saver("ProyectoNumeros/savedweights_capa2")

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

        # def backward(self, X, y_true, y_pred):
        # grad_loss = self.loss_function.compute_gradient(y_true, y_pred)
        # grad_a2 = self.activation2.backward(grad_loss, y_pred)
        # grad_z2 = self.capa2.backward(grad_a2, self.learning_rate)
        # grad_a1 = self.activation1.backward(grad_z2)
        # self.capa1.backward(grad_a1, self.learning_rate)
                
                