#!/usr/bin/evn python3
"""
conducts forward propagation using Dropout
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    X is a numpy.ndarray of shape (nx, m)
    - containing the input data for the network.
    nx is the number of input features
    m is the number of data points
    weights is a dictionary of the weights and biases of the neural network
    L the number of layers in the network
    keep_prob is the probability that a node will be kept
    Returns a dict containing the outputs of each layer
    """
    output = {}
    output['A0'] = X
    for lay in range(L):
        bias = weights['b' + str(lay + 1)]
        A = output['A' + str(lay)]
        W = weights['W' + str(lay + 1)]
        z = np.matmul(W, A) + bias
        if lay == (L - 1):
            t = (np.exp(z) / np.sum(np.exp(z), axis=0))
        else:
            t = np.tanh(z)
            dropout = np.random.binomial(n=1, p=keep_prob, size=t.shape)
            output['D' + str(lay + 1)] = dropout
            t = (t * dropout) / keep_prob
        output['A' + str(lay + 1)] = t
    return output
