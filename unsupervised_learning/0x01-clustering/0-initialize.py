#!/usr/bin/env python3
"""
contains function initialize()
"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means
    """
    if not isinstance(k, int) or k <= 0:
        return None
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    n, d = X.shape
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    return np.random.uniform(low=min_val, high=max_val, size=(k, d))
