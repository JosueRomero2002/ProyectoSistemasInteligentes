import numpy as np
from DenseLayer import DenseLayer
from Activation import ReLUActivation, SoftmaxActivation
from LossFunctions import CrossEntropyLoss

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, 
                 optimizer=None, usingSecondLayer=False, usingLossRegulation=False):
        
        self.usingSecondLayer = usingSecondLayer
        self.usingLossRegulation = usingLossRegulation
        
        # Capa 1 (Oculta)
        self.capa1 = DenseLayer(input_size, hidden_size)
        self.capa1.weights_loader("ProyectoNumeros/savedweights_capa1")
        self.activation1 = ReLUActivation()

        # Capa 2 (Oculta opcional)
        if self.usingSecondLayer:
            self.capa2 = DenseLayer(hidden_size, hidden_size)
            self.capa2.weights_loader("ProyectoNumeros/savedweights_capa2")
            self.activation2 = ReLUActivation()
        
        # Capa 3 (Salida)
        output_input_size = hidden_size if self.usingSecondLayer else hidden_size
        self.capa3 = DenseLayer(output_input_size, output_size)
        self.capa3.weights_loader("ProyectoNumeros/savedweights_capa3")
        self.activation3 = SoftmaxActivation()

        self.loss_function = CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        self.training_loss = []
        self.test_accuracy = []

    def forward(self, X):
        # Propagación hacia adelante
        x = self.capa1.forward(X)
        x = self.activation1.forward(x)
        
        if self.usingSecondLayer:
            x = self.capa2.forward(x)
            x = self.activation2.forward(x)
        
        x = self.capa3.forward(x)
        return self.activation3.forward(x)

    def backward(self, X, y_true, y_pred):
        lambda_l2 = 0.0001 if self.usingLossRegulation else 0.0
        
        # Propagación hacia atrás
        grad = self.loss_function.compute_gradient(y_true, y_pred)
        grad = self.activation3.backward(grad, y_pred)
        grad = self.capa3.backward(grad, lambda_l2)
        
        if self.usingSecondLayer:
            grad = self.activation2.backward(grad)
            grad = self.capa2.backward(grad, lambda_l2)
            
        grad = self.activation1.backward(grad)
        self.capa1.backward(grad, lambda_l2)

    def train(self, X, y, epochs, batch_size, ytest, X_test, saveandprinteach):
        num_samples = X.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, num_samples, batch_size):
                batch_X = X[indices[i:i+batch_size]]
                batch_y = y[indices[i:i+batch_size]]

                y_pred = self.forward(batch_X)
                loss = self.loss_function.compute_loss(batch_y, y_pred)
                self.backward(batch_X, batch_y, y_pred)
                
                epoch_loss += loss
                num_batches += 1

                if self.optimizer is not None:
                    self.optimizer.pre_update_params()
                    self.optimizer.update_params(self.capa1)
                    if self.usingSecondLayer:
                        self.optimizer.update_params(self.capa2)
                    self.optimizer.update_params(self.capa3)
                    self.optimizer.post_update_params()
            
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            self.training_loss.append(avg_epoch_loss)
            
            y_test_pred = self.predict(X_test)
            accuracy = np.mean(np.argmax(ytest, axis=1) == y_test_pred)
            self.test_accuracy.append(accuracy)

            if epoch % saveandprinteach == 0:
                print(f"Epoch [{epoch}] Loss: {avg_epoch_loss:.4f} | Accuracy: {accuracy*100:.2f}%")
                self._save_weights()

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def _save_weights(self):
        self.capa1.weights_saver("ProyectoNumeros/savedweights_capa1")
        if self.usingSecondLayer:
            self.capa2.weights_saver("ProyectoNumeros/savedweights_capa2")
        self.capa3.weights_saver("ProyectoNumeros/savedweights_capa3")

        
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
                
                