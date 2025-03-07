import numpy as np

while(True):

    # Funcion de Transferencia
    class CapaDensa:
        def __init__(self, entradas, neuronas):
            self.pesos = np.random.rand(entradas, neuronas) * 0.01
            self.sesgos = np.zeros((1, neuronas))
        
        def forward(self, datos):
            self.entrada = datos  # Guardamos la entrada para usar en backward
            self.salida = np.dot(datos, self.pesos) + self.sesgos
        
        def backward(self, dSalida, lr=0.01):
            dPesos = np.dot(self.entrada.T, dSalida)
            dSesgos = np.sum(dSalida, axis=0, keepdims=True)
            self.pesos -= lr * dPesos
            self.sesgos -= lr * dSesgos
            return np.dot(dSalida, self.pesos.T)  # Propagar error a la capa anterior

    # Funcion de Activacion
    class ReLU:
        def forward(self, datos):
            self.entrada = datos  # Guardamos la entrada para usar en backward
            self.salida = np.maximum(0, datos)
        
        def backward(self, dSalida):
            return dSalida * (self.entrada > 0)

    class Sigmoide:
        def forward(self, datos):
            self.entrada = datos  # Guardamos la entrada para usar en backward
            self.salida = 1 / (1 + np.exp(-datos))
        
        def backward(self, dSalida):
            return dSalida * (self.salida * (1 - self.salida))

    # Preparar datos
    palindromos = ["oso", "reconocer", "anilina", "radar", "neuquen", "ana", "oso"]
    no_palindromos = ["casa", "perro", "computadora", "python", "redes"]

    X_text = palindromos + no_palindromos
    y_labels = np.array([1] * len(palindromos) + [0] * len(no_palindromos)).reshape(-1, 1)

    # Convertir palabras a representaciones numéricas
    max_length = max(len(word) for word in X_text)
    char_to_int = {char: idx for idx, char in enumerate(set(''.join(X_text)), 1)}
    X_numerico = [np.array([char_to_int[c] for c in word] + [0] * (max_length - len(word))) for word in X_text]
    X_numerico = np.array(X_numerico)

    # Definir la red neuronal
    capa1 = CapaDensa(entradas=max_length, neuronas=8)
    activacion1 = ReLU()
    capa2 = CapaDensa(entradas=8, neuronas=1)
    activacion2 = Sigmoide()

    # Entrenar la red neuronal
    lr = 0.01  # Tasa de aprendizaje
    for _ in range(1000):  # Epochs
        # Forward pass
        capa1.forward(X_numerico)
        activacion1.forward(capa1.salida)
        capa2.forward(activacion1.salida)
        activacion2.forward(capa2.salida)
        
        # Calcular error y backpropagation
        error = activacion2.salida - y_labels
        dSalida2 = activacion2.backward(error)
        dSalida1 = capa2.backward(dSalida2, lr)
        capa1.backward(activacion1.backward(dSalida1), lr)

    # Evaluación
    nueva_palabra =  input("Ingrese una palabra: ")
    nueva_numerica = np.array([char_to_int.get(c, 0) for c in nueva_palabra] + [0] * (max_length - len(nueva_palabra)))
    nueva_numerica = nueva_numerica.reshape(1, -1)

    capa1.forward(nueva_numerica)
    activacion1.forward(capa1.salida)
    capa2.forward(activacion1.salida)
    activacion2.forward(capa2.salida)

    print(f'La palabra "{nueva_palabra}" es un palíndromo con probabilidad: {activacion2.salida[0][0]:.4f}')