import numpy as np

# Funcion de Transferencia
class CapaDensa:
    #                    columnas filas
    def __init__(self, entradas, neuronas):
        self.pesos = np.random.rand(entradas, neuronas) * 0.01
        self.sesgos = np.zeros((1, neuronas))
        
        # .zeros((1, neuronas))

    def forward(self, datos):
        self.salida = np.dot(datos, self.pesos) + self.sesgos


# Funcion de Activacion
class ReLU: # Rectificador Lineal Unitario
    def forward(self, datos: list[float]) -> None:
        self.salida = np.maximum(0, datos)

class Softmax:   
    def forward(self, datos: list[float]) -> None:
        exponencial = np.exp(datos - np.max(datos))
        self.salida = exponencial / np.sum(exponencial, axis=1)

# Clasificacion Binaria
class Sigmoide:
    def forward(self, datos: list[float]) -> None:
        self.salida = 1 / (1 + np.exp(-datos))


# 1, keepdims=True