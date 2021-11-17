#!/usr/bin/env python3
"""
contains function pca()
"""
import numpy as np


def pca(X, ndim):
    """
    performs pca on dataset
    """
    X_avg = X - np.mean(X, axis=0)
    U, S, var = np.linalg.svd(X_avg)
    W = var.T
    W2 = W[:, :ndim]
    return np.dot(X_avg, W2)
