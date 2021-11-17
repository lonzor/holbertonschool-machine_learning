#!/usr/bin/env python3
"""
contains function pca()
"""
import numpy as np


def pca(X, var=0.95):
    """
    performs PCA on dataset
    """
    U, S, vh = np.linalg.svd(X)
    cum_var = np.cumsum(S) / np.sum(S)
    r = np.argwhere(cum_var >= var)[0, 0]
    w = vh[:r + 1].T
    return w
