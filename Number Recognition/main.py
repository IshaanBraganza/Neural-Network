# A model that recognizes numbers


# note that the number recognition repo is now gone, and this looks like it can recognise 10x10 grid of some kind of drawing... 
# I guess we will never know what happened...

import numpy as np
RNG = np.random.default_rng(0)

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)

def init_weights(input_size, output_size):
    return RNG.normal(loc=0.0, scale=np.sqrt(2/input_size), size=(input_size, output_size))

def cost_func(pred, target):
    x = (pred - target) ** 2
    return x.sum() / x.shape[0]

def forward_pass(X, w1, b1, w2, b2, w3, b3):
    z1 = X @ w1 + b1
    a2 = np.maximum(0,  z1)
    z2 = a2 @ w2 + b2
    a3 = np.maximum(0, z2)
    z3 = a3 @ w3 + b3
    output = softmax(z3)
    return output, a2, a3

def back_prop(w2, w3, a2, a3, output, y, X):
    g = 2 * (output - y) / X.shape[0]
    delta_out = output * (g - (g * output).sum(axis=1, keepdims=True))
    w3bp = a3.T @ delta_out
    b3bp = delta_out.sum(axis=0, keepdims=True)

    delta_3 = (delta_out @ w3.T) * (a3 > 0)
    w2bp = a2.T @ delta_3
    b2bp = delta_3.sum(axis=0, keepdims=True)

    delta_2 = (delta_3 @ w2.T) * (a2 > 0)
    w1bp = X.T @ delta_2
    b1bp = delta_2.sum(axis=0, keepdims=True)

    return w3bp, b3bp, w2bp, b2bp, w1bp, b1bp

def training_loop(w1, w2, w3, b1, b2, b3, step):
    for i in range(2500):
        # 100 inputs to 2 hidden layers of 16 neurons each, then to 3 outputs
        output, a2, a3 = forward_pass(X, w1, b1, w2, b2, w3, b3)

        cost = cost_func(output, y)

        w3bp, b3bp, w2bp, b2bp, w1bp, b1bp = back_prop(w2, w3, a2, a3, output, y, X)

        w1 -= w1bp * step
        b1 -= b1bp * step
        w2 -= w2bp * step
        b2 -= b2bp * step
        w3 -= w3bp * step
        b3 -= b3bp * step
    return w1, b1, w2, b2, w3, b3

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

w1, b1, w2, b2, w3, b3 = training_loop(w1, w2, w3, b1, b2, b3, 0.5)

train_out = forward_pass(X, w1, b1, w2, b2, w3, b3)[0]
print((train_out.argmax(1) == y.argmax(1)).mean())
test_out = forward_pass(X20, w1, b1, w2, b2, w3, b3)[0]
print((test_out.argmax(1) == y20.argmax(1)).mean())   # y20 = your 20 targets