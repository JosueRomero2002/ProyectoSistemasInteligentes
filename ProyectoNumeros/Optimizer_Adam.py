import numpy as np

class Optimizer_Adam:
    """
    Docuemtacion Realizada Con Ayuda de DEEPSEEK
    Implementación del optimizador Adam (Adaptive Moment Estimation).
    
    Características:
    - Combina ventajas de RMSProp y Momentum
    - Mantiene tasas de aprendizaje adaptativas por parámetro
    - Incluye corrección de bias para estimaciones iniciales
    - Soporta decaimiento de tasa de aprendizaje
    """
    
    def __init__(self, learning_rate, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        """
        Inicializa el optimizador con parámetros de configuración.
        
        Args:
            learning_rate (float): Tasa de aprendizaje inicial (η)
            decay (float): Tasa de decaimiento para learning rate (default: 0)
            epsilon (float): Pequeño valor para estabilidad numérica (default: 1e-7)
            beta_1 (float): Factor de decaimiento para primer momento (gradientes) (default: 0.9)
            beta_2 (float): Factor de decaimiento para segundo momento (gradientes al cuadrado) (default: 0.999)
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1  # Para el promedio de los gradientes (primer momento)
        self.beta_2 = beta_2  # Para el promedio de los gradientes al cuadrado (segundo momento)

    def pre_update_params(self):
        """
        Actualiza la tasa de aprendizaje antes de los updates de parámetros.
        Aplica decaimiento según: η = η₀ * (1. / (1. + decay * iterations))
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        """
        Actualiza los parámetros de la capa usando el algoritmo Adam.
        
        Args:
            layer (DenseLayer): Capa a actualizar con los gradientes calculados
            
        Pasos:
            1. Calcula momentos actualizados con decaimiento exponencial
            2. Aplica corrección de bias a los momentos
            3. Actualiza parámetros usando momentos adaptativos
        """
        # Verificar existencia de atributos necesarios
        if not hasattr(layer, 'weight_momentums'):
            # Inicializar momentos y cachés si no existen
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Actualizar momentos para pesos (primer y segundo momento)
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        
        # Calcular momentos con corrección de bias (contra el bias inicial)
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        
        # Actualizar cachés (segundo momento no centrado)
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        
        # Calcular cachés con corrección de bias
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        
        # Actualización de parámetros con normalización adaptativa
        layer.weights -= self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases -= self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        """
        Actualiza contador de iteraciones después de actualizar parámetros.
        """
        self.iterations += 1