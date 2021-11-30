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
        cent_copy = np.copy(centroid)
        c_ext = centroid[:, np.newaxis]
        distances = np.sqrt(((X - c_ext) ** 2).sum(axis=2))
        clss = np.argmin(distances, axis=0)
        for j in range(k):
            if X[clss == j].size == 0:
                centroid[j] = np.random.uniform(min_val, max_val, size=(1, d))
            else:
                centroid[j] = X[clss == j].mean(axis=0)

        c_ext = centroid[:, np.newaxis]
        distances = np.sqrt(((X - c_ext) ** 2).sum(axis=2))
        clss = np.argmin(distances, axis=0)
        if (cent_copy == centroid).all():
            break

        return centroid, clss
