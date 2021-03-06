#!/usr/bin/env python3
"""
contains function marginal()
"""
import numpy as np
intersection = __import__('1-intersection').intersection


def marginal(x, n, P, Pr):
    """
    calculates marginal of obtaining data
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

    calc_sum = np.sum(Pr)
    if not np.isclose(calc_sum, 1):
        raise ValueError("Pr must sum to 1")

    intersec = intersection(x, n, P, Pr)
    marg = np.sum(intersec)

    return marg
