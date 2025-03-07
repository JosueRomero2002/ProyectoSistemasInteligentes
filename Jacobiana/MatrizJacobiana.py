

tamanoMatriz = 4

valores = [0.032, 0.087, 0.2369, 0.6439]

tamanoMatriz = len(valores)

for i in range(0, tamanoMatriz):
    for j in range(0, tamanoMatriz):
        if i == j:
            print(valores[i] * (1 - valores[i]), end = " ")
        else:
            print(-valores[i] * valores[j], end = " ")
    print()







