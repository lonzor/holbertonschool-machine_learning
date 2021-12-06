#!/usr/bin/env python3
"""
contains function backward()
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    performs the backward algorithm for a HMM
    """
    if type(Observation) is not np.ndarray or Observation.ndim != 1:
        return None, None
    if type(Emission) is not np.ndarray or Emission.ndim != 2:
        return None, None
    if type(Transition) is not np.ndarray or Transition.ndim != 2:
        return None, None
    if type(Initial) is not np.ndarray or Initial.ndim != 2:
        return None, None

    T = Observation.shape[0]
    N, M = Emission.shape
    B = np.zeros((N, T))
    B[:, T - 1] = np.ones((N))

    if ((Transition.shape[0] != N or Transition.shape[1] != N
         or Initial.shape[0] != N)):
        return None, None

    for x in range(T - 2, -1, -1):
        for y in range(N):
            trans = Transition[y, :]
            ems = Emission[:, Observation[x + 1]]
            B[y, x] = np.sum((trans * B[:, x + 1]) * ems)

    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])
    return P, B
