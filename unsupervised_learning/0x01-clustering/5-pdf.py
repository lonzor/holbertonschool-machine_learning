#!/usr/bin/env python3
"""
contains function pdf()
"""
import numpy as np


def pdf(X, m, S):
    """
    calculates the probability denseity function of a Gaussian dist
    """
    if type(X) is not np.ndarray or type(m) is not np.ndarray:
        return None
    if type(S) is not np.ndarray:
        return None
    if len(X.shape) != 2 or len(S.shape) != 2:
        return None
    if len(m.shape) != 1:
        return None

    n, d = X.shape
    if m.shape[0] != d or S.shape[0] != d or S.shape[1] != d:
        return None

    mean = X - m[None, :]
    det = np.linalg.det(S)
    inv = np.linalg.inv(S)
    norm = 1 / (np.sqrt((((2*np.pi)**d)) * det))
    result = np.exp(-0.5 * np.sum(((mean @ inv) * mean), axis=1))
    pdf = (norm * result)
    P = np.maximum(pdf, 1e-300)

    return P
