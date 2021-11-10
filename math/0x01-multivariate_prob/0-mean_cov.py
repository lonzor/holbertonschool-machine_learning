#!/usr/bin/env python3
"""
contains function mean_cov()
"""
import numpy as np


def mean_cov(X):
    """
    calculates mean and covariance
    """
    dimensions = X.shape[1]

    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    mean = np.zeros((1, dimensions))
    mean[0] = np.mean(X, axis=0)
    covar = np.dot(X.T, X - mean) / (X.shape[0] - 1)
    return mean, covar
