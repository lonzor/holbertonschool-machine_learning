#!/usr/bin/env python3
"""
contains function kmeans()
"""
import sklearn.cluster


def kmeans(X, k):
    """
    performs k-means on a dataset
    """
    k_mean = sklearn.cluster.KMeans(n_clusters=k)
    k_mean.fit(X)
    return k_mean.cluster_centers_, k_mean.labels_
