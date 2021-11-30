#!/usr/bin/env python3
"""
Contains function optimum_k()
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmin) != int or kmin <= 0 or kmin >= X.shape[0]:
        return None, None
    if type(kmax) != int or kmax <= 0 or kmax > X.shape[0]:
        return None, None
    if type(iterations) != int or iterations <= 0:
        return None, None

    results = []
    d_vars = []

    for i in range(kmin, kmax + 1):
        C, clss = kmeans(X, i, iterations)
        results.append((C, clss))
        if i == kmin:
            min_vari = variance(X, C)
        d_vars.append(min_vari - variance(X,C))

    return results, d_vars
