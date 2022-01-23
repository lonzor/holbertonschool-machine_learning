#!/usr/bin/env python 3
"""
contains function bi_rnn()
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    performs forward propagation for a bidirectional RNN
    """
    t, m, i = X.shape
    _, h = h_0.shape
    H_forw = np.zeros((t + 1, m, h))
    H_back = np.zeros((t + 1, m, h))
    H_forw[0] = h_0
    H_back[t] = h_t

    for i in range(t):
        H_forw[i + 1] = bi_cell.forward(H_forw[i], X[i])

    for j in range(t - 1, -1, -1):
        H_back[j] = bi_cell.backward(H_back[j + 1], X[j])

    H = np.concatenate((H_forw[1:], H_back[0:t]), axis=2)
    Y = bi_cell.output(H)

    return H, Y
