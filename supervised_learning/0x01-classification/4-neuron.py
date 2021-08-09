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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        X is a numpy.ndarray with shape (nx, m). Contains input data.
        nx is the number of input features to the neuron
        m is the number of examples
        """
        z = np.matmul(self.W, X) + self.b
        self.__A = 1 / (1 + (np.exp(-z)))
        return (self.A)

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Y is a numpy.ndarray with shape (1, m)
        A is a numpy.ndarray with shape (1, m)
        """
        m = Y.shape[1]
        loss = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        cost = (1 / m) * (-loss)
        return (cost)

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions
        X is a numpy.ndarray with shape (nx, m)
        Y is a numpy.ndarray with shape (1, m)
        nx is the number of input features to the neuron
        m is the number of examples
        """
        predict = self.forward_prop(X)
        cost = self.cost(Y, predict)
        result = predict.round().astype(int)
        return result, cost
