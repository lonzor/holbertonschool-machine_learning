#!/usr/bin/env python3
"""
contains function variance()
"""
import numpy as np


def variance(X, C):
    """
    calculates the total intra-cluster for a data set
    """
    if type(X) is not np.ndarray or type(C) is not np.ndarray:
        return None
    if len(X.shape) != 2 or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    vari = np.sum((X - C[:, np.newaxis])**2, axis=-1)
    val_mean = np.sqrt(vari)
    val_min = np.min(val_mean, axis=0)
    vari = np.sum(val_min ** 2)
    result = np.sum(vari)
    return result
