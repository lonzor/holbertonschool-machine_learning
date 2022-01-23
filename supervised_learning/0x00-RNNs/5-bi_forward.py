#!/usr/bin/env python3
"""
Contains Class BidirectionalCell
"""
import numpy as np


class BidirectionalCell:
    """
    class for bidirectional of RNN
    """

    def __init__(self, i, h, o):
        """
        constructor for class
        """
        self.Whb = np.random.normal(size=(i + h, h))
        self.Whf = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(2 * h, o))
        self.bhb = np.zeros((1, h))
        self.bhf = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        does forward prop
        """
        cat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(cat, self.Whf) + self.bhf)

        return h_next
