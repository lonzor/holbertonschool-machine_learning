#!/usr/bin/env python3
"""
contains convolusional neural network for forward prop function
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    performs forward propagation over a convolutional layer
    - of a neural network
    """
    m, h, w, c = A_prev.shape
    kern_h, kern_w, c, cn = W.shape
    sh, sw = stride

    if padding == 'same':
        pad_h = (((h - 1) * sh) + kern_h - h) // 2
        pad_w = (((w - 1) * sw) + kern_w - w) // 2

    if padding == 'valid':
        pad_h = 0
        pad_w = 0

    pad_img = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                     mode='constant', constant_values=0)
    next_h = (h - kern_h + 2 * pad_h) // sh + 1
    next_w = (w - kern_w + 2 * pad_w) // sw + 1
    result = np.zeros((m, next_h, next_w, cn))

    for i in range(next_h):
        for j in range(next_w):
            for k in range(cn):
                result[:, i, j, k] = np.sum(
                    pad_img[:, i * sh:(kern_h + (i * sh)),
                            j * sw:(kern_w + (j * sw))]
                    * W[:, :, :, k],
                    axis=(1, 2, 3)
                )
    return result
