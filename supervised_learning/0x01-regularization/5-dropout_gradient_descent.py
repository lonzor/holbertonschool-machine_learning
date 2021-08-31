#!/usr/bin/env python3
"""
updates the weights of a neural network with Dropout regularization
- using gradient descent.
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Y is a one-hot numpy.ndarray of shape (classes, m)
    - that contains the correct labels for the data.
    classes is the number of classes
    m is the number of data points
    weights is a dictionary of the weights and biases of the neural network
    cache is a dictionary of the outputs
    - dropout masks of each layer of the neural network.
    alpha is the learning rate
    keep_prob is the probability that a node will be kept
    L is the number of layers of the network
    """
    m = Y.shape[1]
    for lay in range(L, 0, -1):
        A = cache['A' + str(lay)]
        A_prev = cache['A' + str(lay - 1)]
        bias = weights['b' + str(lay)]
        if lay == L:
            dz = (A - Y)
        else:
            og_dz = dz
            dropout = cache['D' + str(lay)]
            dz = np.matmul(w.T, og_dz) * (1 - (A**2))
            dz = (dz * dropout) / keep_prob
        w = weights['W' + str(lay)]
        dw = (1/m) * (np.matmul(dz, A_prev.T))
        db = (1/m) * np.sum(dz, axis=1, keepdims=True)
        weights['W' + str(lay)] = w - (alpha * dw)
        weights['b' + str(lay)] = bias - (alpha * db)
