#!/usr/bin/env python3
"""
contains function markov_chain()
"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Function determines the probability of a markov chain being in a
    particular state after a specified number of iterations:
    """
    if type(P) is not np.ndarray or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if type(s) is not np.ndarray or s.ndim != 2:
        return None
    if s.shape[0] != 1 or s.shape[1] != P.shape[0]:
        return None

    for x in range(t):
        s = np.dot(s, P)

    return s
