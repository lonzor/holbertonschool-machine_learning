#!/usr/bin/env python3
"""
contains function gmm()
"""
import sklearn.mixture


def gmm(X, k):
    """
    Calculates a GMM from a dataset
    """
    mix = sklearn.mixture.GaussianMixture(n_components=k)
    param = mix.fit(X)
    m = param.means_
    S = param.covariances_
    pi = param.weights_
    clss = mix.predict(X)
    bay = mix.bic(X)
    return pi, m, S, clss, bay
