#!/usr/bin/env python3
"""
contains function agglomerative()
"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    performs agglomerative clustering on a dataset
    """
    link = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(link, t=dist,
                                            criterion='distance')
    plt.figure()
    scipy.cluster.hierarchy.dendrogram(link, color_threshold=dist)
    plt.show()

    return clss
