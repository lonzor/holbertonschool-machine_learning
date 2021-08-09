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

        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        prvs = nx
        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            w = "W{}".format(i + 1)
            b = "b{}".format(i + 1)
            self.weights[w] = np.random.randn(layers[i], prvs)\
                * np.sqrt(2 / prvs)
            self.weights[b] = np.zeros((layers[i], 1))
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
