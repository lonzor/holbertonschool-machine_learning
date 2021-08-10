#!/usr/bin/env python3
"""
defines Neuron class
"""
import numpy as np
import matplotlib.pyplot as plt


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

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        X is a numpy.ndarray with shape (nx, m)
        Y is a numpy.ndarray with shape (1, m)
        A is a numpy.ndarray with shape (1, m)
        alpha is the learning rate
        Updates the private attributes __W and __b
        """
        m = Y.shape[1]
        dz = A - Y
        db = np.sum(dz) / m
        dw = np.matmul(X, dz.T) / m
        self.__b = self.__b - (alpha * db)
        self.__W = self.__W - (alpha * dw.T)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the neuron
        X is a numpy.ndarray with shape (nx, m)
        Y is a numpy.ndarray with shape (1, m)
        iterations is the number of iterations to train over
        alpha is the learning rate
        Updates the private attributes __W, __b, and __A
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        lst = []
        x_plot = np.arange(0, iterations + 1, step)

        for i in range(iterations):
            self.__A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
            if i % step == 0 and verbose is True:
                cost = self.cost(Y, self.__A)
                print("Cost after {} iterations: {}".format(i, cost))
            if i % step == 0 and graph is True:
                cost = self.cost(Y, self.__A)
                lst.append(cost)
        if verbose is True:
            cost = self.cost(Y, self.__A)
            print("Cost after {} iterations: {}".format(iterations, cost))
        if graph is True:
            cost = self.cost(Y, self.__A)
            lst.append(cost)
            plt.plot(x_plot, lst, 'b')
            plt.title("Training Cost")
            plt.ylabel("cost")
            plt.xlabel("iteration")
            plt.show()
        return self.evaluate(X, Y)
