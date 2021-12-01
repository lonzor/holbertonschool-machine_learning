#!/usr/bin/env python3
"""
Contains function maximization()
"""
import numpy as np


def maximization(X, g):
    """
    calculates the maximization step in the EM algo for a GMM
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    gauss = g
    k = gauss.shape[0]
    n, d = X.shape

    p_prob = np.sum(gauss, axis=0)
    signal = np.sum(p_prob)
    if signal != X.shape[0]:
        return None, None, None

    pi = np.zeros((k,))
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))
    for j in range(k):
        num1 = np.sum((gauss[j, :, np.newaxis] * X), axis=0)
        num2 = np.sum(gauss[j], axis=0)
        m = num1 / num2

        x = X - m[j]
        sig1 = np.matmul(gauss[j] * x.T, x)
        sig2 = np.sum(gauss[j])
        S[j] = sig1 / sig2

        pi[j] = np.sum(gauss[j]) / n
        maximization = pi, m, S

        return maximization
