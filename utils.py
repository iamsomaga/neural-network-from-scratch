# utils.py
import numpy as np

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Loss function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
