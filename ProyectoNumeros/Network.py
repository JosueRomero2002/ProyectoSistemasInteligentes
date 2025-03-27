import numpy as np
from DenseLayer import DenseLayer
from Activation import ReLUActivation, SoftmaxActivation
from LossFunctions import CrossEntropyLoss

class NeuralNetwork:
    """
    Docuemtacion Realizada Con Ayuda de DEEPSEEK
    Implementa una red neuronal feedforward para clasificación multiclase.
    
    Arquitectura básica:
    Input -> DenseLayer1 -> ReLU -> [DenseLayer2 -> ReLU] -> OutputLayer -> Softmax
    
    Args:
        input_size (int): Número de características de entrada (ej. 784 para MNIST)
        hidden_size (int): Neuronas en la(s) capa(s) oculta(s)
        output_size (int): Neuronas en capa de salida (clases)
        learning_rate (float): Tasa de aprendizaje para SGD (default: 0.1)
        optimizer (Optimizer_Adam): Instancia de optimizador (opcional)
        usingSecondLayer (bool): Habilita segunda capa oculta (default: False)
        usingLossRegulation (bool): Habilita regularización L2 (default: False)
    """
    
    def __init__(self, input_size, hidden_size, output_size, model, learning_rate=0.1, 
                 optimizer=None, usingSecondLayer=False, usingLossRegulation=False):
        """
        Inicializa la red neuronal con parámetros configurables.
        """
        # Configuración de arquitectura
        self.usingSecondLayer = usingSecondLayer
        self.usingLossRegulation = usingLossRegulation
        self.model = model + ("_2" if usingSecondLayer else "") 


        
        
        # Capa 1: Entrada -> Oculta
        self.capa1 = DenseLayer(input_size, hidden_size)
        self.capa1.weights_loader(f"ProyectoNumeros/models/{self.model}/savedweights_capa1")
        self.activation1 = ReLUActivation()

        # Capa 2 Opcional: Oculta -> Oculta
        if self.usingSecondLayer:
            self.capa2 = DenseLayer(hidden_size, hidden_size)
            self.capa2.weights_loader(f"ProyectoNumeros/models/{self.model}/savedweights_capa2")
            self.activation2 = ReLUActivation()
        
        # Capa de Salida: Oculta -> Salida
        output_input_size = hidden_size if not self.usingSecondLayer else hidden_size
        self.capa3 = DenseLayer(output_input_size, output_size)
        self.capa3.weights_loader(f"ProyectoNumeros/models/{self.model}/savedweights_capa3")
        self.activation3 = SoftmaxActivation()

        # Configuración de entrenamiento
        self.loss_function = CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        # Históricos de entrenamiento
        self.training_loss = []    # Pérdida por época
        self.test_accuracy = []    # Precisión en test

    def forward(self, X):
        """
        Propagación hacia adelante a través de la red.
        
        Args:
            X (numpy.ndarray): Datos de entrada (batch_size x input_size)
            
        Returns:
            numpy.ndarray: Salida después de softmax (probabilidades)
            
        Almacena internamente:
            - Salidas de cada capa para backward pass
        """
        x = self.capa1.forward(X)
        x = self.activation1.forward(x)
        
        if self.usingSecondLayer:
            x = self.capa2.forward(x)
            x = self.activation2.forward(x)
        
        x = self.capa3.forward(x)
        return self.activation3.forward(x)

    def backward(self, X, y_true, y_pred):
        """
        Propagación hacia atrás calculando gradientes.
        
        Args:
            X (numpy.ndarray): Entrada original
            y_true (numpy.ndarray): Etiquetas verdaderas (one-hot)
            y_pred (numpy.ndarray): Predicciones de la red
            
        Calcula:
            - Gradientes de pesos y biases para todas las capas
            - Aplica regularización L2 si está habilitada
        """
        lambda_l2 = 0.0001 if self.usingLossRegulation else 0.0
        
        # Gradiente inicial desde CrossEntropy + Softmax
        grad = self.loss_function.compute_gradient(y_true, y_pred)
        
        # Backprop a través de capas
        grad = self.activation3.backward(grad, y_pred)
        grad = self.capa3.backward(grad, lambda_l2)
        
        if self.usingSecondLayer:
            grad = self.activation2.backward(grad)
            grad = self.capa2.backward(grad, lambda_l2)
            
        grad = self.activation1.backward(grad)
        self.capa1.backward(grad, lambda_l2)

    def train(self, X, y, epochs, batch_size, ytest, X_test, saveandprinteach):
        """
        Entrena la red neuronal con los datos proporcionados.
        
        Args:
            X (numpy.ndarray): Datos de entrenamiento
            y (numpy.ndarray): Etiquetas de entrenamiento (one-hot)
            epochs (int): Número de épocas completas
            batch_size (int): Tamaño de mini-batch
            ytest (numpy.ndarray): Etiquetas de validación
            X_test (numpy.ndarray): Datos de validación
            saveandprinteach (int): Frecuencia de guardado/impresión
            
        Efectos secundarios:
            - Actualiza pesos de la red
            - Guarda pesos periódicamente
            - Actualiza históricos de pérdida y precisión
        """
        num_samples = X.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            epoch_loss = 0
            num_batches = 0
            
            # Entrenamiento por mini-batches
            for i in range(0, num_samples, batch_size):
                batch_X = X[indices[i:i+batch_size]]
                batch_y = y[indices[i:i+batch_size]]

                # Forward pass
                y_pred = self.forward(batch_X)
                loss = self.loss_function.compute_loss(batch_y, y_pred)
                
                # Backward pass
                self.backward(batch_X, batch_y, y_pred)
                
                # Actualización de parámetros
                if self.optimizer is None:
                    self._manual_update_weights()
                else:
                    self.optimizer.pre_update_params()
                    self.optimizer.update_params(self.capa1)
                    if self.usingSecondLayer:
                        self.optimizer.update_params(self.capa2)
                    self.optimizer.update_params(self.capa3)
                    self.optimizer.post_update_params()
                
                epoch_loss += loss
                num_batches += 1

            # Cálculo de métricas
            avg_epoch_loss = epoch_loss / num_batches
            self.training_loss.append(avg_epoch_loss)
            
            # Validación
            y_test_pred = self.predict(X_test)
            accuracy = np.mean(np.argmax(ytest, axis=1) == y_test_pred)
            self.test_accuracy.append(accuracy)

            # Logging y guardado
            if epoch % saveandprinteach == 0:
                print(f"Epoch [{epoch}] Loss: {avg_epoch_loss:.4f} | Accuracy: {accuracy*100:.2f}%")
                self._save_weights()

    def predict(self, X):
        """
        Realiza predicciones sobre nuevos datos.
        
        Args:
            X (numpy.ndarray): Datos de entrada (batch_size x input_size)
            
        Returns:
            numpy.ndarray: Clases predichas (índices)
        """
        return np.argmax(self.forward(X), axis=1)

    def _save_weights(self):
        """Guarda pesos de todas las capas en archivos .npy"""
        self.capa1.weights_saver(f"ProyectoNumeros/models/{self.model}/savedweights_capa1")
        if self.usingSecondLayer:
            self.capa2.weights_saver(f"ProyectoNumeros/models/{self.model}/savedweights_capa2")
        self.capa3.weights_saver(f"ProyectoNumeros/models/{(self.model)}/savedweights_capa3")

    def _manual_update_weights(self):
        """Actualiza pesos manualmente usando SGD simple"""
        lr = self.learning_rate
        
        # Actualizar capa 1
        self.capa1.weights -= lr * self.capa1.dweights
        self.capa1.biases -= lr * self.capa1.dbiases
        
        # Actualizar capa 2 si existe
        if self.usingSecondLayer:
            self.capa2.weights -= lr * self.capa2.dweights
            self.capa2.biases -= lr * self.capa2.dbiases
        
        # Actualizar capa de salida
        self.capa3.weights -= lr * self.capa3.dweights
        self.capa3.biases -= lr * self.capa3.dbiases