#  c. Implement a CrossEntropyLoss Class
#  • Compute Loss: Calculate cross-entropy loss between predictions and one-hot 
# labels.
#  • Compute Gradient: Return the gradient of the loss with respect to the softmax 
# input.


import numpy as np


class CrossEntropyLoss:
    def compute_loss(self, y_true, y_pred):
        # 
        return -np.mean(np.sum(y_true * np.log(y_pred) + ((1 - y_true)*np.log(1-y_pred))))

    def compute_gradient(self, y_true, y_pred):
        return (y_pred - y_true) / y_true.shape[0]