import numpy as np

class ReLUActivation:
    """
    Documentacion Realizada con ayuda de DEEPSEEK
    Implementación de la función de activación ReLU (Rectified Linear Unit).
    
    Propiedades:
        - Forward: max(0, x)
        - Backward: 1 si x > 0, 0 en otro caso
    """
    
    def forward(self, inputs):
        """
        Propagación hacia adelante de ReLU.
        
        Args:
            inputs (numpy.ndarray): Valores de entrada de la capa
            
        Returns:
            numpy.ndarray: Valores después de aplicar ReLU. Mismo shape que inputs
        """
        self.inputs = inputs  # Guardar inputs para el backward pass
        return np.maximum(0, inputs)

    def backward(self, grad_output):
        """
        Propagación hacia atrás de ReLU.
        
        Args:
            grad_output (numpy.ndarray): Gradientes recibidos de la capa posterior
            
        Returns:
            numpy.ndarray: Gradientes ajustados según la derivada de ReLU
        """
        grad = grad_output.copy()  # Copiar para no modificar el gradiente original
        grad[self.inputs <= 0] = 0  # Derivada es 0 para valores <= 0
        return grad

class SoftmaxActivation:
    """
    Docuemtacion Realizada Con Ayuda de DEEPSEEK
    Implementación de la función de activación Softmax.
    
    Propiedades:
        - Forward: Distribución de probabilidad sobre clases
        - Backward: Calcula el gradiente usando la matriz Jacobiana
    """
    
    def forward(self, inputs):
        """
        Propagación hacia adelante de Softmax con estabilidad numérica.
        
        Args:
            inputs (numpy.ndarray): Logits de la capa (outputs antes de activación)
            
        Returns:
            numpy.ndarray: Probabilidades normalizadas (suma=1 por fila)
        """
        # Estabilidad numérica: restar el máximo para evitar overflow
        shifted_inputs = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_values = np.exp(shifted_inputs)
        
        # Normalizar las probabilidades
        sum_exp_values = np.sum(exp_values, axis=1, keepdims=True)
        return exp_values / (sum_exp_values + 1e-9)  # +1e-9 para evitar división por 0

    def backward(self, grad_output, outputs):
        """
        Propagación hacia atrás de Softmax.
        
        Args:
            grad_output (numpy.ndarray): Gradientes de la función de pérdida
            outputs (numpy.ndarray): Salidas del forward pass (probabilidades softmax)
            
        Returns:
            numpy.ndarray: Gradientes calculados usando la matriz Jacobiana
        """

        return grad_output