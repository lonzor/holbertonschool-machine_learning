#!/usr/bin/env python3
"""
contains function correlation()
"""
import numpy as np


def correlation(C):
    """
    finds correlation matrix
    """
    if type(C) != np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2:
        raise ValueError("C must be a 2D square matrix")
    if C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    var = np.diag(1 / np.sqrt(np.diag(C)))
    correl = np.matmul(np.matmul(var, C), var)
    return correl
