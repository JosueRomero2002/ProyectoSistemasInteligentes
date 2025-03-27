import numpy as np
import os

class DenseLayer:
    """
    Docuemtacion Realizada Con Ayuda de DEEPSEEK
    
    Implementa una capa densa (fully connected) para redes neuronales.
    
    Características:
    - Inicialización de pesos con He et al. (2015) para activaciones ReLU
    - Soporte para regularización L2
    - Funcionalidad para guardar/cargar pesos desde disco
    - Cálculo eficiente de gradientes durante backpropagation
    """
    
    def __init__(self, input_size, output_size):
        """
        Inicializa los parámetros de la capa.
        
        Args:
            input_size (int): Número de características de entrada
            output_size (int): Número de neuronas en la capa
            
        Inicializa:
            weights: Matriz de pesos con inicialización He et al.
            biases: Vector de biases inicializados en cero
        """
        # Inicialización He para mejorar convergencia con ReLU
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((1, output_size))  # Bias por neurona
    
    def forward(self, inputs):
        """
        Propagación hacia adelante.
        
        Args:
            inputs (numpy.ndarray): Datos de entrada (batch_size x input_size)
            
        Returns:
            numpy.ndarray: Salida de la capa (batch_size x output_size)
            
        Almacena:
            inputs: Guarda las entradas para usar en backpropagation
        """
        self.inputs = inputs  # Guardar para cálculo de gradientes
        return np.dot(inputs, self.weights) + self.biases
    
    def backward(self, grad_output, lambda_l2=0.0):
        """
        Propagación hacia atrás.
        
        Args:
            grad_output (numpy.ndarray): Gradientes de la capa siguiente
            lambda_l2 (float): Factor de regularización L2 (default: 0)
            
        Returns:
            numpy.ndarray: Gradientes para la capa anterior
            
        Calcula y almacena:
            dweights: Gradientes de los pesos
            dbiases: Gradientes de los biases
        """
        # Gradiente de pesos (input^T * grad_output) + término de regularización
        self.dweights = np.dot(self.inputs.T, grad_output) + lambda_l2 * self.weights
        
        # Gradiente de biases (suma sobre el batch)
        self.dbiases = np.sum(grad_output, axis=0, keepdims=True)
        
        # Gradiente para capa anterior (grad_output * W^T)
        grad_input = np.dot(grad_output, self.weights.T)
        
        return grad_input
    
    def weights_saver(self, path="ProyectoNumeros/models/base/savedweights"):
        """
        Guarda pesos y biases en archivos .npy.
        
        Args:
            path (str): Directorio destino para guardar los archivos
            
        Crea:
            weights.txt.npy: Archivo con matriz de pesos
            biases.txt.npy: Archivo con vector de biases
        """
        if not os.path.exists(path):
            os.makedirs(path)  # Crear directorio si no existe
            
        np.save(f"{path}/weights.txt", self.weights)
        np.save(f"{path}/biases.txt", self.biases)
        print("Weights and Biases saved")
    
    def weights_loader(self, path="ProyectoNumeros/models/base/savedweights"):
        """
        Carga pesos y biases desde archivos .npy.
        
        Args:
            path (str): Directorio con los archivos de pesos
            
        Comportamiento:
            - Si los archivos existen y coinciden las dimensiones: carga los pesos
            - Si hay error de dimensiones: mantiene inicialización aleatoria
            - Si no encuentra archivos: usa inicialización aleatoria
        """
        try:
            # Cargar archivos
            loaded_weights = np.load(f"{path}/weights.txt.npy")
            loaded_biases = np.load(f"{path}/biases.txt.npy")
            
            # Verificar compatibilidad de dimensiones
            if loaded_weights.shape == self.weights.shape and loaded_biases.shape == self.biases.shape:
                self.weights = loaded_weights
                self.biases = loaded_biases
                print(f"Loaded weights from {path}")
            else:
                print(f"Dimension mismatch in {path}. Using random initialization.")
        except FileNotFoundError:
            print(f"No weights found in {path}. Using random initialization.")