
class Codificador:
    def __init__(self):
        self.BancoDeLetras = [ '?', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                         'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                         'v', 'w', 'x', 'y', 'z']

    def codificar(self, palabra):
      
        # Transformar palabras a un arreglo de numeros
        palabra = palabra.lower()
        palabra = palabra.replace(" ", "")

        palabraNumeros = []

        for letra in palabra:
            if letra not in self.BancoDeLetras:
                return "La palabra contiene caracteres no permitidos"
            palabraNumeros.append(self.BancoDeLetras.index(letra))

        for i in range(8-len(palabraNumeros)):
            palabraNumeros.append(0)



        self.palabraNumeros = palabraNumeros
      


# Clasificacion Binaria

# Sigmoide



# palabra = "awiwi"

# if len(palabra) > 8:
#     print("La palabra es muy larga")
#     exit()

# BancoDeLetras = [ '?', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
#                          'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
#                          'v', 'w', 'x', 'y', 'z']
       
# palabra = palabra.lower()
# palabra = palabra.replace(" ", "")

# palabraNumeros = []

# for letra in palabra:
#         palabraNumeros.append(BancoDeLetras.index(letra))

# for i in range(8-len(palabraNumeros)):
#     palabraNumeros.append(0)


# print(palabraNumeros)

