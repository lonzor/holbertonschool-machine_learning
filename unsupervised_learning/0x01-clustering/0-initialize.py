#!/usr/bin/env python3
"""
contains function initialize()
"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means
    """
    if type(X) is not int or type(x) is not np.ndarray:
        return None
    if len(X.shape) != 2 or k < 0:
        return None

    n, d = X.shape
    if k == 0:
        return None
    min_val = np.amin(X, axis=0)
    max_val = np.amax(X, axis=0)
    return np.random.uniform(low, high, size=(k, d))
