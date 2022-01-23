#!/usr/bin/env python3
"""
contains class GRUCell
"""
import numpy as np


class GRUCell:
    """
    class declaration
    """
    def __init__(self, i, h, o):
        """
        constructor for class GRUCell
        """
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.Wz = np.random.normal(size=(h + i, h))
        self.Wr = np.random.normal(size=(h + i, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.Wh = np.random.normal(size=(h + i, h))
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

    def forward(self, h_prev, x_t):
        """
        forward prop
        """
        h_stack = np.hstack((h_prev, x_t))
        z_up = self.sigmoid(h_stack @ self.Wz + self.bz)
        zgate_res = self.sigmoid(h_stack @ self.Wr + self.br)
        stack2 = np.hstack((zgate_res * h_prev, x_t))
        h_tanh = np.tanh(stack2 @ self.Wh + self.bh)
        h_next = (np.ones_like(z_up) - z_up) * h_prev + z_up * h_tanh
        y = self.softmax(h_next @ self.Wy + self.by)

        return h_next, y
