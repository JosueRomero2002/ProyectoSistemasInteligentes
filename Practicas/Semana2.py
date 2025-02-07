# import numpy as np

# np.ra

# np.rad2deg # radianes a grados

# np.deg2rad # grados a radianes

# np.reandom.randn(2,3)

# np.random.randn(3,4)

# np.zeros((1,4))


# # //matmul0

#Import RedesNueronales
import Practicas.Semana21 as rn

capa1 = rn.CapaDensa(4, 3)

capa1.forward([[1, 2, 3, 4,5]])

print(capa1.salida)

capa2 = rn.CapaDensa(3, 3)

capa2.forward(capa1.salida)

print(capa2.salida)





