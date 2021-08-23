#!/usr/bin/env python3
"""Standardizes a matrix"""
import numpy as np


def normalize(X, m, s):
    """
    X is the numpy.ndarray of shape (d, nx) to normalize
    d is the number of data points
    nx is the number of features
    m is a numpy.ndarray of shape (nx,)
    s is a numpy.ndarray of shape (nx,)
    Returns: The normalized X matrix
    """
    norm = (X - m) / s
    return norm
