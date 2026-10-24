import numpy as np

w = np.array([
    [0.2,  0.5],   # w1.1, w1.2
    [-0.4, 0.1],   # w2.1, w2.2
    [0.7, -0.3],   # w3.1, w3.2
    [0.1,  0.8],   # w4.1, w4.2
    [0.3, -0.6],   # w5.1, w5.2
])

z = np.array([
    [1, 0],
    [1, 0],
    [0, 1],
    [1, 0],
    [0, 1]
])


# cost function
x = (w - z) ** 2 
a = x.sum() / x.shape[0]

print(a)
