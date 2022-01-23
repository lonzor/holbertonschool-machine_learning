#!/usr/bin/env python3
"""
contains class GRUCell
"""
import numpy as np


class LSTMCell:
    """
    class declaration
    """
    def __init__(self, i, h, o):
        """
        constructor for class GRUCell
        """
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.Wf = np.random.normal(size=(h + i, h))
        self.Wu = np.random.normal(size=(h + i, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.Wc = np.random.normal(size=(h + i, h))
        self.Wo = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))

    def softmax(self, x):
        """
        performs softmax
        """
        expo = np.exp(x)
        return expo / np.sum(expo, 1, keepdims=True)

    def sigmoid(self, x):
        """
        performs sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, c_prev, x_t):
        """
        forward prop
        """
        h_stack = np.hstack((h_prev, x_t))
        f_sig = self.sigmoid(h_stack @ self.Wf + self.bf)
        u_sig = self.sigmoid(h_stack @ self.Wu + self.bu)
        c_tanh = np.tanh(h_stack @ self.Wc + self.bc)
        c_next = f_sig * c_prev + u_sig * c_tanh
        o_sig = self.sigmoid(h_stack @ self.Wo + self.bo)
        h_next = o_sig * np.tanh(c_next)
        y = self.softmax(h_next @ self.Wy + self.by)

        return h_next, c_next, y
