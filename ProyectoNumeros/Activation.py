#  b. Implement Activation Classes
#  • ReLUActivation:
#  ◦ Forward: Apply ReLU (output = max(0, input)).
#  ◦ Backward: Compute gradient for ReLU input.
#  • SoftmaxActivation:
#  ◦ Forward: Compute softmax probabilities.
#  ◦ Backward: Compute gradient for softmax input


import numpy as np

class ReLUActivation:
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, grad_output):
        grad = grad_output.copy()
        grad[self.inputs <= 0] = 0
        return grad

class SoftmaxActivation:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        sum_exp_values = np.sum(exp_values, axis=1, keepdims=True)
        return exp_values / (sum_exp_values + 1e-9)



    def backward(self, grad_output, outputs):
        return grad_output * outputs * (1 - outputs)




