#!/usr/bin/env python3
"""
contains init and function kernel
contains class GaussianProcess
"""
import numpy as np


class GaussianProcess:
    """
    class representing a Gaussian Process
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Initializes class GaussianProcess
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        calculates the covariance kernel matrix between two matrices
        """
        sq_dist1 = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1)
        sq_dist2 = 2 * np.dot(X1, X2.T)
        sq_dist = sq_dist1 - sq_dist2
        result = self.sigma_f ** 2 * np.exp(0.5 / self.l ** 2 * sq_dist)
        return result
