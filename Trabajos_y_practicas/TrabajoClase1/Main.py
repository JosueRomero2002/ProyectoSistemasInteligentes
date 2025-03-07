import Codificador as cod
import RedNeuronal as rn


while True:
    palabra = input("Ingrese una palabra: ")

    if palabra == "exit":
        break


    codificador = cod.Codificador()
    codificador.codificar(palabra)
    # print(codificador.palabraNumeros)

    capa1 = rn.CapaDensa(8, 16)

    capa1.forward(codificador.palabraNumeros)
    relu1 = rn.ReLU()
    relu1.forward(capa1.salida)



    


    capa2 = rn.CapaDensa(16, 16)

    capa2.forward(capa1.salida)
    relu2 = rn.ReLU()
    relu2.forward(capa2.salida)



    capa3 = rn.CapaDensa(16, 16)
    capa3.forward(capa2.salida)
    relu3 = rn.ReLU()
    relu3.forward(capa3.salida)

    

    for i in range(1000):
        capaOculta = rn.CapaDensa(16, 16)
        capaOculta.forward(capa3.salida)
        reluOculta = rn.ReLU()
        reluOculta.forward(capaOculta.salida)
        capa3 = capaOculta
        relu3 = reluOculta

    



    capaSalida = rn.CapaDensa(16, 4)
    capaSalida.forward(capa3.salida)

    sigmoide = rn.Sigmoide()

    sigmoide.forward(capaSalida.salida)

    print("Salida de la capa de salida con softmax -------------------")
    print(sigmoide.salida)




# softmax = rn.Softmax()
# softmax.forward(capaSalida.salida)

# print("Salida de la capa de salida -------------------")
# print(capaSalida.salida)


# print("Salida de la capa de salida con softmax -------------------")
# print(softmax.salida)
