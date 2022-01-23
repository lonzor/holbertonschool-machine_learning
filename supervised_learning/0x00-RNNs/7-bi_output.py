#!/usr/bin/env python3
"""
Class BidirectionalCell with  contructor, forward(), and backward()
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
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)
        self.bhb = np.zeros((1, h))
        self.bhf = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def softmax(x):
        """
        gets softmax
        """
        expo = np.exp(x)
        return expo / np.sum(expo, -1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        does forward prop
        """
        cat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.matmul(cat, self.Whf) + self.bhf
        h_next = np.tanh(h_next)

        return h_next

    def backward(self, h_next, x_t):
        """
        does backward prop
        """
        cat = np.concatenate((h_next, x_t), axis=1)
        back_prop = np.matmul(cat, self.Whb) + self.bhb
        back_prop = np.tanh(back_prop)

        return back_prop

    def output(self, H):
        """
        finds output of RNN
        """
        Y = self.softmax(H @ self.Wy + self.by)
        return Y
