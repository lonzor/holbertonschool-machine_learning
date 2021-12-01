#!/usr/bin/env python3
"""
contains function kmeans()
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs kmeans on a dataset
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(k) is not int or k <= 0 or k > X.shape[0]:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None

    n, d = X.shape
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    centroid = np.random.uniform(low=min_val, high=max_val, size=(k, d))

    for i in range(iterations):
        distan = np.sqrt(((X - centroid[:, np.newaxis])**2).sum(axis=2))
        clss = np.argmin(distan, axis=0)
        cent_cpy = centroid.copy()
        for c in range(k):
            if len(X[c == clss]) == 0:
                centroid[c] = np.random.uniform(low=np.min(X, axis=0),
                                                high=np.max(X, axis=0),
                                                size=(1, d))
            else:
                centroid[c] = np.mean(X[c == clss], axis=0)
        if np.array_equal(cent_cpy, centroid):
            break

        distan = np.sqrt(((X - centroid[:, np.newaxis])**2).sum(axis=2))
        clss = np.argmin(distan, axis=0)

        return centroid, clss
