#!/usr/bin/env python3
"""
contains function absorbing()
"""
import numpy as np


def absorbing(P):
    """
    determines if a markov chain is absorbing
    """
    if len(P.shape) != 2:
        return None

    num_states1, num_states2 = P.shape

    if (num_states1 != num_states2):
        return None
    if type(P) is not np.ndarray:
        return None

    diag = np.diagonal(P)

    if (diag == 1).all():
        return True
    if not (diag == 1).any():
        return False

    for x in range(num_states1):
        for y in range(num_states2):
            if (x == y) and (x + 1 < len(P)):
                if P[x + 1][y] == 0 and P[x][y + 1] == 0:
                    return False
    return True
