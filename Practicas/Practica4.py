import numpy as np

def mse(y_real, y_pred):
    return np.mean((y_real - y_pred) ** 2)


y_real = np.array([1.5, 2.0, 3.5])
y_pred = np.array([1.4, 2.1, 3.2])

print(mse(y_real, y_pred)) # 0.03333333333333333

def mae(y_real, y_pred):
    return np.mean(np.abs(y_real - y_pred))

print(mae(y_real, y_pred)) 

def cross_entropy(y_real, y_pred):
    return -np.sum(y_real * np.log(y_pred))

y_real = np.array([1, 0, 0])
y_pred = np.array([0.9, 0.1, 0.2])

print(cross_entropy(y_real, y_pred)) # 0.10536051565782628

y_real = np.array([1, 0, 0])
y_pred = np.array([0.1, 0.9, 0.2])

print(cross_entropy(y_real, y_pred)) # 2.3025850929940455


def binary_cross_entropy(y_real, y_pred):
    return -np.mean(y_real * np.log(y_pred) + ((1 - y_real) * np.log(1 - y_pred)))

y_real = np.array([1, 0, 1])
y_pred = np.array([0.9, 0.1, 0.8])

print(binary_cross_entropy(y_real, y_pred)) # 