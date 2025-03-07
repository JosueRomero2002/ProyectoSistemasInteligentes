import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(train_csv, test_csv):
    train_data = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)
    
    X_train = train_data.drop(columns=['label']).values / 255.0
    y_train = train_data['label'].values
    
    X_test = test_data.drop(columns=['label']).values / 255.0 
    y_test = test_data['label'].values
    
    return X_train, y_train, X_test, y_test

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))

def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = softmax(self.z2)
        return self.a2

    def backward(self, X, y_true, y_pred):
        m = X.shape[0]
        
        dz2 = y_pred - y_true
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        dz1 = np.dot(dz2, self.W2.T) * sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X, y, epochs):
        self.loss_history = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            
            loss = cross_entropy_loss(y, y_pred)
            self.loss_history.append(loss)
            
            self.backward(X, y, y_pred)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.show()

X_train, y_train, X_test, y_test = load_data('/kaggle/input/mnist-in-csv/mnist_train.csv', '/kaggle/input/mnist-in-csv/mnist_test.csv')


y_train_encoded = one_hot_encode(y_train, 10)
y_test_encoded = one_hot_encode(y_test, 10)

input_size = 784 
hidden_size = 64 
output_size = 10 
learning_rate = 0.1

nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
nn.train(X_train, y_train_encoded, epochs=5000)

nn.plot_loss()

y_test_pred = nn.predict(X_test)
test_accuracy = accuracy(y_test_encoded, one_hot_encode(y_test_pred, 10))
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


def plot_predictions(X_test, y_test, y_pred, num_samples=10):
    plt.figure(figsize=(10, 10))
    
    for i in range(num_samples):
        idx = np.random.randint(0, X_test.shape[0])
        
        image = X_test[idx].reshape(28, 28)
        true_label = y_test[idx]
        predicted_label = y_pred[idx]
        
        plt.subplot(5, 5, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f"True: {true_label}, Pred: {predicted_label}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

y_test_pred = nn.predict(X_test)

plot_predictions(X_test, y_test, y_test_pred, num_samples=10)