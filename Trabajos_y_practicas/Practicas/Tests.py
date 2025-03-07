# Escribir codigo en python para calcular la salida de una capa con 2 nueronas y 3 entradas. 
# Asumiendo que las 3 entradas estan conectadas a cada neurona.

import numpy as np  

# fila col
# 1 x 3
entradas = [1, 2, 3]

# 3 
pesos = [
    [4,5,6],
    [7,8,9]
]

sesgos = [2,3]

output = []

output = np.dot(pesos, entradas) + sesgos

print(output)

