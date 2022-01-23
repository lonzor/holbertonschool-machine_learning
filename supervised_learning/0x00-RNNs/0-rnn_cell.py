#!/usr/bin/env python3
"""
contains class RNNCell
"""
import numpy as np


class RNNCell:
    """
    the class RNNCell
    """
    def __init__(self, i, h, o):
        """
        constructor for RNNCell class
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def softmax(x):
        """
        activates softmax
        """
        expo = np.exp(x)
        soft = expo / np.sum(expo, 1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        does forward propagation
        """
        cat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.matmul(cat, self.Wh) + self.bh
        h_next = np.tanh(h_next)
        y = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, y
