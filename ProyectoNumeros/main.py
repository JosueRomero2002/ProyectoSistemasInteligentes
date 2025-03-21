from Dataset import MnistDataset
from Network import NeuralNetwork
import numpy as np
from Optimizer_Adam import Optimizer_Adam
import matplotlib.pyplot as plt

mnist_train = MnistDataset()
mnist_train.load("ProyectoNumeros/Mnist/dataset/train-images-idx3-ubyte", "ProyectoNumeros/Mnist/dataset/train-labels-idx1-ubyte")
mnist_test = MnistDataset()
mnist_test.load("ProyectoNumeros/Mnist/dataset/t10k-images-idx3-ubyte", "ProyectoNumeros/Mnist/dataset/t10k-labels-idx1-ubyte")

X_train = mnist_train.get_flattened_data()
y_train = mnist_train.get_one_hot_labels()
X_test = mnist_test.get_flattened_data()
y_test = mnist_test.get_one_hot_labels()


#   <----------MENU
usingSecondLayer = True   # Activar segunda hidden layer
#Warning: Changing this value might rewrite the saved weights due to the different number of weights
usingAdamOptimizer = False
usingLossRegulation = False


# Hyperparameters Settings      
input_size = 784
hidden_size = 128
output_size = 10
learning_rate = 0.001
epochs = 20
batchSize = 64
saveandprinteach = 5 



# TESTING <----------MENU
num_samples=3
showResults = True



# LossRegulation = 1 or 2 (using l1 or l2) (l1 is optional but l2 is obligatory)



optimizer = Optimizer_Adam(learning_rate=learning_rate, decay=1e-3)
# nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate, optimizer)

nn = NeuralNetwork(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    learning_rate=learning_rate,
    optimizer=optimizer if usingAdamOptimizer else None,
    usingSecondLayer=usingSecondLayer,
    usingLossRegulation=usingLossRegulation
)

nn.train(X_train, y_train, epochs, batchSize, y_test, X_test, saveandprinteach)

y_test_pred = nn.predict(X_test)
accuracy = np.mean(np.argmax(y_test, axis=1) == y_test_pred)
print("Accuracy: [" + str(accuracy * 100) + "%]")



# TEST A SAMPLE (RANDOM)
if(showResults):
    def show_sample_predictions(model, X, y, num_samples=num_samples):
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

    show_sample_predictions(nn, X_test, y_test)

    # GRAPHS

    import matplotlib.pyplot as plt

    def plot_training_history(network):
        plt.figure(figsize=(12, 5))
        

        plt.subplot(1, 2, 1)
        plt.plot(network.training_loss, label='Training')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(network.test_accuracy, label='Test', color='orange')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()


    plot_training_history(nn)

