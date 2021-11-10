#!/usr/bin/env python3
"""
contains function mean_cov()
"""
import numpy as np


def mean_cov(X):
    """
    calculates mean and covariance
    """
    if X.shape[0] < 2:
        raiseValueError("X must contain multiple data points")
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    data_points = X.shape[0]
    dimensions = X.shape[1]
    mean = np.mean(X, axis=0).reshape(1, dimensions)
    X = X - mean
    covar = ((np.dot(X.T, X)) / (data_points - 1))
    return mean, covar
