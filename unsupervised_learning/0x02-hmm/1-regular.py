#!/usr/bin/env python3
"""
Contains function regular()
"""
import numpy as np


def regular(P):
    """
    determines the steady state probabilities of a regular markov chain
    """
    if type(P) is not np.ndarray or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if not (P > 0).all():
        return None

    num_state = P.shape[0]
    evals, evecs = np.linalg.eig(P.T)
    not_norm_state = (evecs / evecs.sum())
    norm_state = np.dot(not_norm_state.T, P)
    for x in norm_state:
        if (x >= 0).all() and np.isclose(x.sum(), 1):
            return x.reshape(1, num_state)
