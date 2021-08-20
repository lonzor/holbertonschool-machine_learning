#!/usr/bin/env python3
"""contains method for nomalize constants"""
import numpy as np


def normalization_constants(X):
    """
    X is the numpy.ndarray of shape (m, nx) to normalize
    m is the number of data points
    nx is the number of features
    """
    return np.mean(X, axis=0), np.std(X, axis=0)
