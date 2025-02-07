import Practicas.Semana21 as rn

capa1 = rn.CapaDensa(5, 10)

#                5 entradas
capa1.forward([1, 2, 3, 4, 5 ])

relu1 = rn.ReLU()
relu1.forward(capa1.salida)

print("Pesos de la capa 1 -------------------")
print(capa1.pesos)

print("Salida de la capa 1 -------------------")
print(capa1.salida)

capa2 = rn.CapaDensa(10, 10)
capa2.forward(capa1.salida)

relu2 = rn.ReLU()
relu2.forward(capa2.salida)

print("Pesos de la capa 2 -------------------")
print(capa2.pesos)

print("Salida de la capa 2 -------------------")


print(capa2.salida)


capaSalida = rn.CapaDensa(10, 4)
capaSalida.forward(capa2.salida)

softmax = rn.Softmax()
softmax.forward(capaSalida.salida)

print("Salida de la capa de salida -------------------")
print(capaSalida.salida)


print("Salida de la capa de salida con softmax -------------------")
print(softmax.salida)


