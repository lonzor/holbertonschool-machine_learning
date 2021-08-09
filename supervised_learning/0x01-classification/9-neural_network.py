#!/usr/bin/env python3
"""contains NeuralNetwork class"""
import numpy as np


class NeuralNetwork:
    """
    defines a neural network with one hidden layer
    performing binary classification
    """

    def __init__(self, nx, nodes):
        """
        constructor for class NeuralNetwork
        nx is the number of input features
        nodes is the number of nodes found in the hidden layer
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """getter for weights vector (hidden layer)"""
        return self.__W1

    @property
    def b1(self):
        """getter for bias (hidden layer)"""
        return self.__b1

    @property
    def A1(self):
        """getter for activated output (hidden layer)"""
        return self.__A1

    @property
    def W2(self):
        """getter for weights vector (output neuron)"""
        return self.__W2

    @property
    def b2(self):
        """getter for bias (output neuron)"""
        return self.__b2

    @property
    def A2(self):
        """getter for activated output (prediction)"""
        return self.__A2
