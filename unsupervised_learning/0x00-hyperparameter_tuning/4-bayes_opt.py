#!/usr/bin/env python3
"""
Contains BayesianOptimization class and constructor
"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Class performs a Gaussian Process
    """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """
        constructor for class BayesianOptimization class
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        X_s = np.linspace(bounds[0], bounds[1], num=ac_samples)
        self.X_s = X_s.reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates next best sample location
        """
        mu, sigma = self.gp.predict(self.X_s)
        with np.errstate(divide='warn'):
            improve = mu - self.xsi - np.min(self.gp.Y)
            Z = improve / sigma
            exp_imp = improve * norm.cdf(Z) + sigma + norm.pdf(Z)
            exp_imp[sigma == 0.00] = 0.0

        X_next = self.X_s[np.argmax(exp_imp)]
        return X_next, exp_imp
