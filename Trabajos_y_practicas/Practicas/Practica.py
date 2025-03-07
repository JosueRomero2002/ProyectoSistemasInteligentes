# Escribir codigo en python para calcular la salida de una capa con dos nueronas y tres entradas. 
# Asumiendo que las 3 entradas estan conectadas a cada neurona.

import Practicas2 as capaDensa


capa1 = capaDensa(2,3)

datosInicales = [[1, 2], [3,4], [5,6]]
capa1.forward(datosInicales)

capa2 = capaDensa(3,2)

capa2.forward(capa1.output)


capa3 = capaDensa(2,1)

capa3.forward(capa2.output)

print(capa3.output)