

import numpy as np


#  2. Build the Neural Network Components
#  a. Implement a DenseLayer Class
#  • Constructor: Initialize weights and biases for a layer.
#  ◦ Inputs: number_of_neurons, input_size.
#  ◦ Weights: Randomly initialize with small values (e.g., np.random.randn() * 
# 0.01).
#  ◦ Biases: Initialize as zeros.


class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))



#  • Forward Method: 
# ◦ Compute output = input @ weights + biases.

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases

        #  1 
# • Backward Method:
#  ◦ Compute gradients for weights, biases, and input using the chain rule.
#  ◦ Return gradient for the input to pass to the previous layer.
#  • Update Method: 
# ◦ Adjust weights and biases using gradient descent

    def backward(self, grad_output, learning_rate):
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(self.inputs.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0, keepdims=True)
        
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases
        
        return grad_input
