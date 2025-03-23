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
        # Compute Jacobian matrix for softmax
        batch_size, num_classes = outputs.shape
        grad = np.empty_like(grad_output)
        for i in range(batch_size):
            softmax_vector = outputs[i].reshape(-1, 1)
            jacobian_matrix = np.diagflat(softmax_vector) - np.dot(softmax_vector, softmax_vector.T)
            grad[i] = np.dot(jacobian_matrix, grad_output[i])
        return grad
