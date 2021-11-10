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
