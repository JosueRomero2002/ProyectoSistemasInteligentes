from Dataset import MnistDataset
from Network import NeuralNetwork
import numpy as np
from Optimizer_Adam import Optimizer_Adam
import matplotlib.pyplot as plt
import time 

# Docuemtacion Realizada Con Ayuda de DEEPSEEK

# Carga del dataset MNIST
mnist_train = MnistDataset()
mnist_train.load(
    "ProyectoNumeros/Mnist/dataset/train-images-idx3-ubyte",  # Ruta imágenes entrenamiento
    "ProyectoNumeros/Mnist/dataset/train-labels-idx1-ubyte"   # Ruta etiquetas entrenamiento
)
mnist_test = MnistDataset()
mnist_test.load(
    "ProyectoNumeros/Mnist/dataset/t10k-images-idx3-ubyte",   # Ruta imágenes test
    "ProyectoNumeros/Mnist/dataset/t10k-labels-idx1-ubyte"    # Ruta etiquetas test
)

# Preparación de datos
X_train = mnist_train.get_flattened_data()  # Imágenes aplanadas (60000x784)
y_train = mnist_train.get_one_hot_labels()  # Etiquetas one-hot (60000x10)
X_test = mnist_test.get_flattened_data()    # Datos de test (10000x784)
y_test = mnist_test.get_one_hot_labels()    # Etiquetas test (10000x10)

# Configuración del modelo (MENU DE OPCIONES)
usingSecondLayer = True    # Activar segunda capa oculta (aumenta capacidad del modelo)
usingAdamOptimizer = True  # Usar Adam en lugar de SGD manual
usingLossRegulation = True # Aplicar regularización L2

# Direccion de Modelo
model = "model1"

# Hiperparámetros del modelo
input_size = 784        # 28x28 pixeles (MNIST)
hidden_size = 128      # Neuronas en capa oculta
output_size = 10        # 10 dígitos (0-9)
learning_rate = 0.001     # Tasa de aprendizaje inicial
epochs = 500           # Iteraciones completas sobre el dataset
batchSize = 64      # Tamaño del mini-batch
saveandprinteach = 1    # Frecuencia de guardado/impresión (cada X épocas)

# Configuración de pruebas
num_samples = 3        # Número de ejemplos a mostrar
showResults = True     # Mostrar gráficos al finalizar

# Inicialización del optimizador Adam (si está activado)
optimizer = Optimizer_Adam(
    learning_rate=learning_rate, 
    decay=1e-3  # Decaimiento de tasa de aprendizaje
)

# Marca Inicial de Tiempo
start_time = time.time()  


# Creación de la red neuronal
nn = NeuralNetwork(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    model=model,
    learning_rate=learning_rate,
    optimizer=optimizer if usingAdamOptimizer else None,  # Selección dinámica de optimizador
    usingSecondLayer=usingSecondLayer,
    usingLossRegulation=usingLossRegulation
    
)

# Entrenamiento del modelo
nn.train(
    X_train, y_train, 
    epochs=epochs,
    batch_size=batchSize,
    ytest=y_test,       # Etiquetas de test para validación
    X_test=X_test,      # Datos de test para validación
    saveandprinteach=saveandprinteach  # Frecuencia de guardado
)

# Marca Final de Entrenamiento
end_time = time.time() 
training_time = end_time - start_time 

# Evaluación final del modelo
y_test_pred = nn.predict(X_test)
accuracy = np.mean(np.argmax(y_test, axis=1) == y_test_pred)
print("Accuracy: [" + str(accuracy * 100) + "%]")
print(f"                     ≈ {training_time/60:.2f} minutos")

# Visualización de resultados (si está activado)
if showResults:
    def show_sample_predictions(model, X, y, num_samples=num_samples):
        """Muestra predicciones de muestra con imágenes"""
        indices = np.random.choice(len(X), num_samples)
        
        for i in indices:
            image = X[i].reshape(28, 28)
            prediction = model.predict(X[i].reshape(1, -1))[0]
            true_label = np.argmax(y[i])
            
            plt.figure(figsize=(4, 4))
            plt.imshow(image, cmap='gray')
            plt.title(f"Real: {true_label} | Pred: {prediction}")
            plt.axis('off')
            plt.show()

    # Mostrar ejemplos aleatorios
    show_sample_predictions(nn, X_test, y_test)

    # Visualización del historial de entrenamiento
    def plot_training_history(network):
        """Genera gráficos de pérdida y precisión durante el entrenamiento"""
        plt.figure(figsize=(12, 5))
        
        # Gráfico de pérdida
        plt.subplot(1, 2, 1)
        plt.plot(network.training_loss, label='Training')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Gráfico de precisión
        plt.subplot(1, 2, 2)
        plt.plot(network.test_accuracy, label='Test', color='orange')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    plot_training_history(nn)