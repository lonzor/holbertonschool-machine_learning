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
        mu, _ = self.gp.predict(self.gp.X)
        mu2, sigma = self.gp.predict(self.X_s)
        if self.minimize:
            mu3 = np.min(mu)
        else:
            mu3 = np.max(mu)

        improve = mu3 - mu2 - self.xsi
        Z = improve / sigma
        exp_imp = ((improve * norm.cdf(Z)) + (sigma * norm.pdf(Z)))
        exp_imp[sigma == 0.0] = 0.0
        X_next = self.X_s[np.argmax(exp_imp)]

        return X_next, np.array(exp_imp)

    def optimize(self, iterations=100):
        """
        optimizes the black-box function
        """
        for itr in range(iterations):
            optim_x, _  = self.acquisition()
            if optim_x in self.gp.X:
                break
            optim_y = self.f(optim_x)
            self.gp.update(optim_x, optim_y)

        pos = np.argmin(self.gp.Y)
        optim_x = self.gp.X[pos]
        optim_y = np.array(self.gp.Y[pos])

        return optim_x, optim_y
