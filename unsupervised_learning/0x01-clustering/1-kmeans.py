#!/usr/bin/env python3
"""
contains function kmeans()
"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means
    """
    if not isinstance(k, int) or k <= 0:
        return None
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    n, d = X.shape
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    return np.random.uniform(low=min_val, high=max_val, size=(k, d))


def kmeans(X, k, iterations=1000):
    """
    Performs kmeans on a dataset
    """
    if type(iterations) is not int or iterations < 1:
        return None, None
    centroids = initialize(X, k)
    if centroids is None:
        return None, None
    num_manip = None
    while iterations:
        iterations = iterations - 1
        cent_copy = centroids.copy()
        arr_manip = np.apply_along_axis(np.subtract, 1, X, centroids)
        arr_manip = np.argmin(np.square(arr_manip).sum(axis=2), axis=1)
        for c in range(centroids.shape[0]):
            X_where = np.argwhere(arr_manip == c)
            if X_where.shape[0] == 0:
                centroids[c] = initialize(X, 1)
            else:
                centroids[c] = np.mean(X[X_where], axis=0)
        if np.all(cent_copy == centroids):
            return centroids, arr_manip
    arr_manip = np.apply_along_axis(np.subtract, 1, X, centroids)
    arr_manip = np.argmin(np.square(arg_manip).sum(axis=2), axis=1)
    return centroids, arr_manip
