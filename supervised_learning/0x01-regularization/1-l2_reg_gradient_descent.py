#!/usr/bin/env python3
"""
updates the weights and biases of a neural network
using gradient descent with L2 regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Y is a one-hot numpy.ndarray of shape (classes, m)
    - that contains the correct labels for the data.
    classes is the number of classes
    m is the number of data points
    weights is a dictionary of the weights and biases of the neural network
    cache is a dictionary of the outputs of each layer of the neural network
    alpha is the learning rate
    lambtha is the L2 regularization parameter
    L is the number of layers of the network
    """
    m = Y.shape[1]
    for lay in range(L, 0, -1):
        A = cache['A' + str(lay)]
        A_prev = cache['A' + str(lay - 1)]
        if lay == L:
            dz = (A - Y)
        else:
            og_dz = dz
            dz = np.matmul(W.T, og_dz) * (1 - (A**2))
        W = weights['W' + str(lay)]
        b = weights['b' + str(lay)]
        dw = (1 / m) * (np.matmul(dz, A_prev.T)) + ((lambtha / m) * W)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        weights['W' + str(lay)] = W - (alpha * dw)
        weights['b' + str(lay)] = b - (alpha * db)
