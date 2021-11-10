#!/usr/bin/env python3
"""
contains class MultiNormal
"""
import numpy as np


class MultiNormal():
    """
    class multinormal
    """

    def __init__(self, data):
        """
        constructor for class
        """
        if type(data) is not np.ndarray:
            raise TypeError("data must be a 2D numpy.ndarray")
        if len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        mean = np.mean(data, axis=1).reshape((data.shape[0], 1))
        self.mean = mean
        self.cov = np.matmul(data - self.mean, data.T) / (data.shape[1] - 1)

    def pdf(self, x):
        """
        calculates the PDF at a data point
        """
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        if (len(x.shape) != 2 or x.shape[1] != 1):
            raise ValueError("x must have the shape ({}, 1")
        if x.shape[0] != self.mean.shape[0]:
            raise ValueError("x must have the shape({}, 1")

        d = np.sqrt(((2 * np.pi) ** x.shape[0]) * np.linalg.det(self.cov))
        c = np.linalg.inv(self.cov)
        e = (-0.5 * np.matmul(np.matmul((x - self.mean).T, c), x - self.mean))
        pdf = (1 / d) * np.exp(e[0][0])
        return pdf
