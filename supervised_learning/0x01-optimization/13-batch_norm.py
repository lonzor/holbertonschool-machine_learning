#!/usr/bin/env python3
"""
normalizes an unactivated output of a neural network
using batch normalization
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Z is a numpy.ndarray of shape (m, n) that should be normalized
    m is the number of data points
    n is the number of features in Z
    gamma is a numpy.ndarray of shape (1, n)
    beta is a numpy.ndarray of shape (1, n)
    epsilon is a small number used to avoid division by zero
    Returns: the normalized Z matrix
    """
    var = np.var(Z, axis=0)
    m = np.mean(Z, axis=0)
    norm = (Z - m) / (np.sqrt(var + epsilon))
    return gamma * norm + beta
