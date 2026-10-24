# A model that recognizes shapes in images.
import numpy as np
RNG = np.random.default_rng(0)
from shapes_dataset import X, y

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)

def init_weights(input_size, output_size):
    return RNG.normal(loc=0.0, scale=np.sqrt(2/input_size), size=(input_size, output_size))

def cost_func(pred, target):
    x = (pred - target) ** 2
    return x.sum() / x.shape[0]

def learn():
    pass

# Hidden Layer 1 (100 -> 16)
w1 = init_weights(100, 16)
# [[16 weights] * 100 inputs] = 100x16 matrix

# Hidden Layer 2 (16 -> 16)
w2 = init_weights(16, 16)
# [[16 weights] * 16 inputs] = 16x16 matrix

# Output Layer (16 -> 3)
w3 = init_weights(16, 3)
# [[3 weights] * 16 inputs] = 16x3 matrix

# Biases are safe to initialize to zero
b1 = np.zeros((1, 16))
b2 = np.zeros((1, 16))
b3 = np.zeros((1, 3))

# 100 inputs to 2 hidden layers of 16 neurons each, then to 3 outputs

a2 = np.maximum(0, X @ w1 + b1)
a3 = np.maximum(0, a2 @ w2 + b2)
output = softmax(a3 @ w3 + b3)

cost = cost_func(output, y)