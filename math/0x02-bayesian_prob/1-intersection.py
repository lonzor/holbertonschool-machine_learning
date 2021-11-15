#!/usr/bin/env python3
"""
contains function liklihood()
"""
import numpy as np


def intersection(x, n, P, Pr):
    """
    calculates likelihood of obtaining data
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if (not isinstance(P, np.ndarray)) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or (Pr.shape != P.shape):
        raise TypeError(
            "Pr must be a numpy.ndarray with the same shape as P")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose([np.sum(Pr)], [1.])[0]:
        raise ValueError("Pr must sum to 1")

    numer = np.math.factorial(n)
    denom = np.math.factorial(x) * np.math.factorial(n - x)
    fact = numer / denom
    likely = fact * (np.power(P, x)) * (np.power((1 - P), (n - x)))
    intersect = likely * Pr

    return intersect
