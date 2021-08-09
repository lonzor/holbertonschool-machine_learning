#!/usr/bin/env python3
"""Contains method that is a one-hot encoder"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    converts a numeric label vector into a one-hot matrix
    Y is a numpy.ndarray with shape (m,)
    m is the number of examples
    classes is the maximum number of classes found in Y
    Returns: a one-hot encoding of Y with shape (classes, m)
    Or None on failure
    """
    if type(Y) is not np.ndarray or type(classes) is not int:
        return None
    if classes < 2 or classes < np.amax(Y):
        return None

    hot_encoded = np.zeros((classes, len(Y)))
    hot_encoded[Y, np.arange(len(Y))] = 1
    return hot_encoded
