
#  c. Implement a CrossEntropyLoss Class
#  • Compute Loss: Calculate cross-entropy loss between predictions and one-hot 
# labels.
#  • Compute Gradient: Return the gradient of the loss with respect to the softmax 
# input

# CrossEntropyLoss Implemented as LossFunctions
# FIX: Change this afterwards

import numpy as np

class CrossEntropyLoss:
    def compute_loss(self, y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))

    def compute_gradient(self, y_true, y_pred):
        return y_pred - y_true




# l1w = lambda_l1w * sum(abs(weights))
# l2w = lambda_l2w * sum(weights**2)

# loss = data_loss + l1w + l2w





# l2w = lambda_l2w * sum(weights**2)

# loss = data_loss + l2w