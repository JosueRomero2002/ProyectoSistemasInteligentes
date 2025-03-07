import numpy as np

class capaDensa:
    def __init__(self, Num_entradas, Num_neuronas):
        self.pesos = np.random.rand(Num_entradas, Num_neuronas) * 0.01
        self.sesgos = np.zeros((1, Num_neuronas))

    def forward(self, entradas):
        self.output = np.dot(entradas, self.pesos) + self.sesgos

