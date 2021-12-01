#!/usr/bin/env python3
"""
contains function expectation()
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None

    n, d = X.shape
    k = pi.shape[0]
    if d != m.shape[1] or d != S.shape[1] or d != S.shape[2]:
        return None, None
    if k != m.shape[0] or k != S.shape[0]:
        return None, None

    cent_mean = m
    covar = S
    gauss = np.zeros((k, n))

    for j in range(k):
        like_hood = pdf(X, cent_mean[j], covar[j])
        prev = pi[j]
        gauss[j] = like_hood * prev
    g = gauss / np.sum(gauss, axis=0)
    log_like = np.sum(np.log(np.sum(gauss, axis=0)))

    return g, log_like
