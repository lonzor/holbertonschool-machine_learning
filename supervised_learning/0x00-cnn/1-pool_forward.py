#!/usr/bin/env python3
"""
Contains  function pool_forward
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    performs forward propagation over a pooling layer of a neural network
    """
    m, h, w, c = A_prev.shape
    kern_h, kern_w = kernel_shape
    sh, sw = stride
    ph = (h - kern_h) // sh + 1
    pw = (w - kern_w) // sw + 1
    result = np.zeros((m, ph, pw, c))

    if mode == 'max':
        func = np.amax
    if mode == 'avg':
        func = np.average

    for i in range(ph):
        for j in range(pw):
            result[:, i, j, :] = func(
                A_prev[:, i*sh: i*sh + kern_h, j*sw: j*sw + kern_w, :],
                axis=(1, 2)
            )
    return result
