#!/usr/bin/env python3
"""
contains function viterbi()
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    calculates the most likely sequence of hidden states for a HMM
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

    if ((Transition.shape[0] != N or Transition.shape[1] != N
         or Initial.shape[0] != N)):
        return None, None

    T = Observation.shape[0]
    rev_pnt = np.zeros((N, T))
    A = np.zeros((N, T))
    A[:, 0] = Initial.T * Emission[:, Observation[0]]

    for x in range(1, T):
        for y in range(N):
            trans = Transition[:, y]
            ems = Emission[y, Observation[x]]
            A[y, x] = np.amax(trans * A[:, x - 1] * ems)
            rev_pnt[y, x - 1] = np.argmax(trans * A[:, x - 1] * ems)

    arr = [0 for i in range(T)]
    s = np.argmax(A[:, T - 1])
    arr[0] = s

    rev_idx = 1
    for i in range(T - 2, -1, -1):
        arr[rev_idx] = int(rev_pnt[int(s), i])
        s = rev_pnt[int(s), i]
        rev_idx = rev_idx + 1

    arr.reverse()
    P = np.amax(A[:, T - 1], axis=0)

    return arr, P
