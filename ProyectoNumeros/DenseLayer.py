import numpy as np
import os



class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((1, output_size))
    
    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases
    
    def backward(self, grad_output, lambda_l2=0.0):
        self.dweights = np.dot(self.inputs.T, grad_output) + lambda_l2 * self.weights
        self.dbiases = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.weights.T)
        return grad_input
    
    def weights_saver(self, path="ProyectoNumeros/savedweights"):
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(f"{path}/weights.txt", self.weights)
        np.save(f"{path}/biases.txt", self.biases)
        print("Weights and Biases saved")
    
    def weights_loader(self, path="ProyectoNumeros/savedweights"):
        try:
            loaded_weights = np.load(f"{path}/weights.txt.npy")
            loaded_biases = np.load(f"{path}/biases.txt.npy")
            
            if loaded_weights.shape == self.weights.shape and loaded_biases.shape == self.biases.shape:
                self.weights = loaded_weights
                self.biases = loaded_biases
                print(f"Loaded weights from {path}")
            else:
                print(f"Dimension mismatch in {path}. Using random initialization.")
        except FileNotFoundError:
            print(f"No weights found in {path}. Using random initialization.")
