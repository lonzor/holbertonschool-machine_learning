#!/usr/bin/env python3
"""Contains class DeepNeuralNetwork"""
import numpy as np


class DeepNeuralNetwork:
    """
    defines a deep neural network performing binary classification
    """

    def __init__(self, nx, layers):
        """
        constructor for DeepNeuralNetwork class
        nx is the number of input features
        layers is a list representing the number of nodes
        in each layer of the network
        L: The number of layers in the neural network.
        cache: A dictionary to hold values of the network.
        weights: A dictionary to hold all weights and bias
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        prvs = nx
        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            w = "W{}".format(i + 1)
            b = "b{}".format(i + 1)
            self.__weights[w] = np.random.randn(layers[i], prvs)\
                * np.sqrt(2 / prvs)
            self.__weights[b] = np.zeros((layers[i], 1))
            prvs = layers[i]

    @property
    def L(self):
        """
        getter for number of layers in the neural network.
        """
        return self.__L

    @property
    def cache(self):
        """
        getter for dictionary
        holds all intermediary values of the network
        """
        return self.__cache

    @property
    def weights(self):
        """
        getter for dictionary
        holds all weights and bias of the network
        """
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        X is a numpy.ndarray with shape (nx, m)
        nx is the number of input features to the neuron
        m is the number of examples
        Updates the private attribute __cache
        Returns the output of the neural network and the cache
        """
        self.__cache["A0"] = X
        for i in range(self.L):
            b = self.__weights["b{}".format(i + 1)]
            W = self.__weights["W{}".format(i + 1)]
            z = np.matmul(W, self.cache["A{}".format(i)]) + b
            A = 1 / (1 + (np.exp(-z)))
            self.__cache["A{}".format(i + 1)] = A
        return (A, self.__cache)

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
        A, predict = self.forward_prop(X)
        cost = self.cost(Y, A)
        result = A.round().astype(int)
        return result, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        """
        m = Y.shape[1]
        b_prp = {}
        for idx in range(self.L, 0, -1):
            A1 = cache["A{}".format(idx - 1)]
            if idx == self.L:
                b_prp["dz{}".format(idx)] = (cache["A{}".format(idx)] - Y)
            else:
                A2 = cache["A{}".format(idx)]
                dz_prvs = b_prp["dz{}".format(idx + 1)]
                b_prp["dz{}".format(idx)] = \
                    (np.matmul(W.T, dz_prvs)*(A2*(1 - A2)))
            W = self.weights["W{}".format(idx)]
            dz = b_prp["dz{}".format(idx)]
            dw = np.matmul(dz, A1.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            self.__weights["W{}".format(idx)] = W - (alpha * dw)
            self.__weights["b{}".format(idx)] = \
                self.weights["b{}".format(idx)] - (alpha * db)
