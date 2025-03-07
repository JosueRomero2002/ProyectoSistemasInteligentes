import numpy as np

class CapaDensa:
    def __init__(self, entradas, neuronas):
        print("===================CAPA DENSA GENERADA=======================")



        print("Pesos: ")
        self.pesos = np.array([[1,2,3],
        
                                [3,4, 5]])



        print(self.pesos.T)


        self.sesgos = np.array( [ 1, 2])
        print("Sesgos: ")
        print(self.sesgos)

        print("=============================================================")

    def forward(self, entradas):
        self.salida = np.dot(entradas, self.pesos.T)  + self.sesgos

    
# Pesos de 5 entradas y 8 neuronas
# [[0.00379745 0.00296325 0.00467972 0.00642249 0.00136421 0.00651933 0.00777699 0.00844669]
#  [0.00980703 0.00282747 0.00489691 0.00102854 0.00759635 0.00071262  0.00898731 0.00456585]
#  [0.00469795 0.001415   0.00510555 0.00709179 0.00394555 0.00540572   0.00690695 0.00409009]
#  [0.00620411 0.00126034 0.00248868 0.00661428 0.00139049 0.00602146  0.00925159 0.00830899]
#  [0.00153425 0.00746243 0.00102104 0.00702418 0.00014406 0.00565339  0.00096446 0.00940101]]


# Pesos: 
# [[0.0098699  0.00304659 0.00674186]
#  [0.00628795 0.00252434 0.00638886]]
Entrada1 = 3
Neuronas1 = 2
Capa1 = CapaDensa(Entrada1, Neuronas1)

Entradas = np.array([[ 3,2,1]]) # Ahora tiene forma (1,3)

Capa1.forward(Entradas)

print(Capa1.salida)

# (1, 3)  (3,2)   = ( 1, 2)


Capa2 = CapaDensa(2, 1)



EntradasManual = [
    3,2,1
]

PesosManual = [
 [1,2, 3],
 [3,4, 5],

]

SesgosManual = [ 1, 2]



print("Compresion de Listas==========================")

print([( sum((i*w) for i, w in zip(EntradasManual, pesos)) + sesgos   ) for pesos, sesgos in zip(PesosManual, SesgosManual) ])


for pesos, sesgos in zip(PesosManual, SesgosManual):

    print(sum([(i*w) for i, w in zip(EntradasManual, pesos)]) + sesgos)
 

print("===============================================")

transPuesta = [ [ 0 for x in range(len(PesosManual))] for y in range (len(PesosManual[0]))]





for x in range(len(PesosManual)):

    for y in range(len(PesosManual[0])):
        transPuesta[y][x] = PesosManual[x][y]


print([fila for fila in PesosManual])
trandpuestaGod = [ [fila[i] for fila in PesosManual] for i in range (len(PesosManual[0]))]


print(PesosManual)
print(transPuesta)

print(trandpuestaGod)
print("===============================================")


SalidaManual = [
(EntradasManual[0] * PesosManual[0][0] +
EntradasManual[1] * PesosManual[0][1] +
EntradasManual[2] * PesosManual[0][2] ) + SesgosManual[0]
,

(EntradasManual[0] * PesosManual[1][0] +
EntradasManual[1] * PesosManual[1][1] +
EntradasManual[2] * PesosManual[1][2] ) + SesgosManual[1]

]

print(SalidaManual)





print([i for i in range(10, 0, -1)])
