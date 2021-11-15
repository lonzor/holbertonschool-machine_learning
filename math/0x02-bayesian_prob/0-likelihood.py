#!/usr/bin/env python3
"""
contains function liklihood()
"""
import numpy as np


def likelihood(x, n, P):
    """
    calculates likelihood of obtaining data
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        message = "x must be an integer that is greater than or equal to 0"
        raiseValueError(message)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if (not isinstance(P, np.ndarray)) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray.")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    numer = np.math.factorial(n)
    denom = np.math.factorial(x) * np.math.factorial(n - x)
    fact = numer / denom
    likely = fact * (np.power(P, x)) * (np.power((1 - P), (n - x)))

    return likely
