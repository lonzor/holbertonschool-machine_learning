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
        moo, sig = self.gp.predict(self.X_s)
        sig = sig.reshape(-1, 1)
        with np.errstate(divide='warn'):
            if self.minimize:
                opt_moo = np.min(self.gp.Y)
                imp = (opt_moo - moo - self.xsi).reshape(-1, 1)
            else:
                opt_moo = np.amax(self.gp.Y)
                imp = (moo - opt_moo - self.xsi).reshape(-1, 1)
            Z = imp / sig
            exp_imp = imp * norm.cdf(Z) + sig * norm.pdf(Z)
            exp_imp[sig == 0.0] = 0.0
        X_next = self.X_s[np.argmax(exp_imp)]
        return (X_next, exp_imp.reshape(-1))

    def optimize(self, iterations=100):
        """
        optimizes the black-box function
        """
        for itr in range(iterations):
            optim_x, _ = self.acquisition()
            if optim_x in self.gp.X:
                break
            optim_y = self.f(optim_x)
            self.gp.update(optim_x, optim_y)

        pos = np.argmin(self.gp.Y)
        optim_x = self.gp.X[pos]
        optim_y = np.array(self.gp.Y[pos])

        return optim_x, optim_y
