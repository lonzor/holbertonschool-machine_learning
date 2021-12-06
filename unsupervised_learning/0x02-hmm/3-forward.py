#!/usr/bin/env python3
"""
contains function forward()
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    performs forward algo for a HMM
    """
    if type(Observation) is not np.ndarray or Observation.ndim != 1:
        return None, None
    if type(Emission) is not np.ndarray or Emission.ndim != 2:
        return None, None
    if type(Transition) is not np.ndarray or Transition.ndim != 2:
        return None, None
    if type(Initial) is not np.ndarray or Initial.ndim != 2:
        return None, None

    N, M = Emission.shape
    T = Observation.shape[0]

    if Transition.shape[0] != N or Transition.shape[1] != N:
        return None, None
    if Initial.shape[0] != N:
        return None, None

    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]

    for x in range(1, T):
        for y in range(N):
            trans = Transition[:, y]
            ems = Emission[y, Observation[x]]
            F[y, x] = np.sum(trans * F[:, x - 1] * ems)
    P = np.sum(F[:, -1])
    return P, F
