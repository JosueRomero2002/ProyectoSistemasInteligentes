# Escribir codigo en python para calcular la salida de una capa con dos nueronas y tres entradas. 
# Asumiendo que las 3 entradas estan conectadas a cada neurona.

import Practicas.Practicas2 as capaDensa

capaDensa = capaDensa.capaDensa

capa1 = capaDensa(10,5)

datosInicales = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                 [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                 [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                 [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                 [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]]
capa1.forward(datosInicales)

capa2 = capaDensa(5,2)

capa2.forward(capa1.output)


capa3 = capaDensa(2,1)

capa3.forward(capa2.output)

print(capa3.output)