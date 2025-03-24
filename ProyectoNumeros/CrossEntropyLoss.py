import numpy as np

class CrossEntropyLoss:
    """
    Docuemtacion Realizada Con Ayuda de DEEPSEEK
    
    Implementa la pérdida de Entropía Cruzada para problemas de clasificación.
    
    Características clave:
    - Diseñada para usarse con la activación Softmax en la última capa
    - Calcula la pérdida promedio sobre el batch
    - Incluye estabilidad numérica con epsilon (1e-9)
    - Proporciona gradientes eficientes para backpropagation
    """
    
    def compute_loss(self, y_true, y_pred):
        """
        Calcula la pérdida de entropía cruzada entre predicciones y etiquetas reales.
        
        Args:
            y_true (numpy.ndarray): Etiquetas reales en formato one-hot (batch_size x num_classes)
            y_pred (numpy.ndarray): Predicciones de la red (probabilidades) (batch_size x num_classes)
            
        Returns:
            float: Valor promedio de la pérdida en el batch
            
        Notas:
            - Usa logaritmos naturales (base e)
            - Añade un pequeño epsilon (1e-9) para evitar log(0)
        """
        # Versión numéricamente estable con clipping de valores
        # La implementación asume que y_pred viene de una capa softmax
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))
    
    def compute_gradient(self, y_true, y_pred):
        """
        Calcula el gradiente de la pérdida con respecto a las salidas de softmax.
        
        Args:
            y_true (numpy.ndarray): Etiquetas reales one-hot (batch_size x num_classes)
            y_pred (numpy.ndarray): Predicciones de la red (batch_size x num_classes)
            
        Returns:
            numpy.ndarray: Gradientes de la pérdida (mismo shape que y_pred)
            
        Notas:
            - La derivada se simplifica cuando se usa con softmax (y_pred - y_true)
            - El gradiente se normaliza por el tamaño del batch
        """
        # La derivada combinada softmax + cross entropy tiene esta forma simplificada
        return (y_pred - y_true) / y_true.shape[0]