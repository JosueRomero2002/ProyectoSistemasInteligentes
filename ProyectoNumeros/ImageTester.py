import numpy as np
import matplotlib.pyplot as plt
from Network import NeuralNetwork

def load_trained_model():
    """
    Docuemtacion Realizada Con Ayuda de DEEPSEEK
    Carga un modelo pre-entrenado desde los pesos guardados.
    
    Returns:
        NeuralNetwork: Modelo neuronal con pesos cargados
        
    Notas:
        - Asume que los pesos están guardados en los directorios:
          - ProyectoNumeros/savedweights_capa1
          - ProyectoNumeros/savedweights_capa2
        - Crea una nueva instancia de red neuronal con la misma arquitectura que durante el entrenamiento
    """
    nn = NeuralNetwork(input_size=784, hidden_size=128, output_size=10)
    nn.capa1.weights_loader("ProyectoNumeros/savedweights_capa1")
    nn.capa2.weights_loader("ProyectoNumeros/savedweights_capa2")
    return nn

def process_custom_image(image_path):
    """
    Preprocesa una imagen personalizada para ser compatible con el modelo MNIST.
    
    Args:
        image_path (str): Ruta del archivo de imagen a procesar
        
    Returns:
        numpy.ndarray: Imagen procesada (1x784) normalizada
        
    Pasos de procesamiento:
        1. Conversión a escala de grises
        2. Redimensionamiento a 28x28 pixeles
        3. Normalización de valores [0-255] a [0.0-1.0]
        4. Aplanado a vector 1D (784 elementos)
    """
    from PIL import Image  # Importación local para reducir dependencias
    
    img = Image.open(image_path).convert('L')  # Convertir a monocromo
    img = img.resize((28, 28))  # Ajustar tamaño a 28x28 pixeles
    img_array = np.array(img)  # Convertir a numpy array
    img_array = img_array.reshape(1, 784) / 255.0  # Aplanar y normalizar
    return img_array

def test_custom_image(model, image_path):
    """
    Realiza una predicción sobre una imagen personalizada y muestra resultados.
    
    Args:
        model (NeuralNetwork): Modelo entrenado cargado
        image_path (str): Ruta de la imagen a evaluar
        
    Returns:
        int: Dígito predicho (0-9)
        
    Muestra:
        - Imagen procesada
        - Título con predicción
    """
    processed_image = process_custom_image(image_path)
    prediction = model.predict(processed_image)
    
    # Visualización de la imagen y predicción
    plt.imshow(processed_image.reshape(28, 28), cmap='gray')
    plt.title(f"Predicción: {prediction[0]}")
    plt.axis('off')
    plt.show()
    
    return prediction[0]

# Ejemplo de uso
if __name__ == "__main__":
    model = load_trained_model()
    result = test_custom_image(model, "ProyectoNumeros/CustomImages/img2.png")
    print("El número identificado es:", result)