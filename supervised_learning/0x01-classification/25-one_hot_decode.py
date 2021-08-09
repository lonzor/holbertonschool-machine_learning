#!/usr/bin/env python3
"""
contains method that decodes a one-hot matrix
"""
import numpy as np


def one_hot_decode(one_hot):
    """
    converts a one-hot matrix into a vector of labels
    one_hot is a one-hot encoded numpy.ndarray with shape (classes, m)
    classes is the maximum number of classes
    m is the number of examples
    """
    if type(one_hot) is not np.ndarray:
        return None
    if one_hot.ndim != 2:
        return None

    hot_decoded = np.argmax(one_hot, axis=0)
    return hot_decoded
