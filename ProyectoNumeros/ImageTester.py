import numpy as np
import matplotlib.pyplot as plt
from Network import NeuralNetwork


def load_trained_model():
    nn = NeuralNetwork(input_size=784, hidden_size=128, output_size=10)
    nn.capa1.weights_loader("ProyectoNumeros/savedweights_capa1")
    nn.capa2.weights_loader("ProyectoNumeros/savedweights_capa2")
    return nn


def process_custom_image(image_path):
    from PIL import Image
    img = Image.open(image_path).convert('L')  # Convertir a escala de grises
    img = img.resize((28, 28))  # Redimensionar a 28x28
    img_array = np.array(img)
    img_array = img_array.reshape(1, 784) / 255.0  # Aplanar y normalizar
    return img_array

# Probar con una imagen
def test_custom_image(model, image_path):
    processed_image = process_custom_image(image_path)
    prediction = model.predict(processed_image)
    
    plt.imshow(processed_image.reshape(28, 28), cmap='gray')
    plt.title(f"Predicción: {prediction[0]}")
    plt.axis('off')
    plt.show()
    
    return prediction[0]

# Uso
model = load_trained_model()
result = test_custom_image(model, "ProyectoNumeros/CustomImages/img2.png")
print("El número identificado es:", result)