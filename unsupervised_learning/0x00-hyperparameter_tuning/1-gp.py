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
        k_matrix = np.exp(-((X1 - X2.T) ** 2) / (2 * (self.l ** 2)))
        result = k_matrix * (self.sigma_f ** 2)
        return result

    def predict(self, X_s):
        """
        predicts mean, standard deviation of points in Gaussian process
        """
        k_s = self.kernel(self.X, X_s)
        k_s2 = self.kernel(X_s, X_s)
        k_inv = np.linalg.inv(self.K)

        mu = k_s.T.dot(k_inv).dot(self.Y).reshape(X_s.shape[0])
        sigma = np.diag(k_s2 - k_s.T.dot(k_inv).dot(k_s))
        return mu, sigma
