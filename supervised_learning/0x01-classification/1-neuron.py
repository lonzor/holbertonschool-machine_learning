#!/usr/bin/env python3
"""
defines Neuron class
"""
import numpy as np


class Neuron:
    """
    Defines a single neuron performing binary classification.
    """

    def __init__(self, nx):
        """constructor for Neuron class"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        getter for __W
        weights vector for the neuron
        """
        return (self.__W)

    @property
    def b(self):
        """
        getter for __b
        The bias for the neuron.
        """
        return (self.__b)

    @property
    def A(self):
        """
        getter for __A
        The activated output of the neuron.
        """
        return (self.__A)
