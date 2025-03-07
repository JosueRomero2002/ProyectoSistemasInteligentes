import CapaDensa as cd


capa1 = cd.CapaDensa(784, 128) 
activacion1 = cd.ReLU()
capa2 = cd.CapaDensa(128, 10) 
activacion2 = cd.Softmax()

