#!/usr/bin/env python3
"""contains NeuralNetwork class"""
import numpy as np
import matplotlib.pyplot as plt


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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        X is a numpy.ndarray with shape (nx, m)
        nx is the number of input features to the neuron
        m is the number of examples
        Updates the private attributes __A1 and __A2
        """
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1/(1 + np.exp(-z1))
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1/(1 + np.exp(-z2))
        return self.__A1, self.__A2

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
        A1, predict = self.forward_prop(X)
        cost = self.cost(Y, predict)
        result = predict.round().astype(int)
        return result, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        X is a numpy.ndarray with shape (nx, m)
        nx is the number of input features to the neuron
        m is the number of examples
        Y is a numpy.ndarray with shape (1, m)
        A1 is the output of the hidden layer
        A2 is the predicted output
        alpha is the learning rate
        Updates the private attributes __W1, __b1, __W2, and __b2
        """
        m = Y.shape[1]
        dz2 = A2 - Y
        dw2 = np.matmul(dz2, A1.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m
        dz1 = np.matmul(self.__W2.T, dz2) * (A1 * (1 - A1))
        dw1 = np.matmul(dz1, X.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m
        self.__W2 = self.__W2 - (alpha * dw2)
        self.__W1 = self.__W1 - (alpha * dw1)
        self.__b2 = self.__b2 - (alpha * db2)
        self.__b1 = self.__b1 - (alpha * db1)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the neural network
        X is a numpy.ndarray with shape (nx, m)
        nx is the number of input features to the neuron
        m is the number of examples
        Y is a numpy.ndarray with shape (1, m)
        iterations is the number of iterations to train over
        alpha is the learning rate
        Updates attributes __W1, __b1, __A1, __W2, __b2, and __A2
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
            self.__A1, self.__A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
            if i % step == 0 and verbose is True:
                cost = self.cost(Y, self.__A2)
                print("Cost after {} iterations: {}".format(i, cost))
            if i % step == 0 and graph is True:
                cost = self.cost(Y, self.__A2)
                lst.append(cost)
        if verbose is True:
            cost = self.cost(Y, self.__A2)
            print("Cost after {} iterations: {}".format(iterations, cost))
        if graph is True:
            cost = self.cost(Y, self.__A2)
            lst.append(cost)
            plt.plot(x_plot, lst, 'b')
            plt.title("Training Cost")
            plt.ylabel("cost")
            plt.xlabel("iteration")
            plt.show()
        return self.evaluate(X, Y)
