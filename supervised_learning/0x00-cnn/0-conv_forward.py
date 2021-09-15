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
    kern_h, kern_w, kc, nc = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = (((h - 1) * sh) + kern_h - h) // 2
        pw = (((w - 1) * sw) + kern_w - w) // 2
    if padding == 'valid':
        ph = 0
        pw = 0
    pad_img = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                     mode='constant', constant_values=0)
    nh = (h + (2 * ph) - kern_h) // sh + 1
    nw = (w + (2 * pw) - kern_w) // sw + 1
    result = np.zeros((m, nh, nw, nc))
    for i in range(nc):
        for j in range(nh):
            for k in range(nw):
                result[:, j, k, i] = np.sum(np.multiply(
                    W[:, :, :, i],
                    pad_img[:, sh*j: sh*j + kern_h, sw*k: sw*k + kern_w]),
                    axis=(1, 2, 3)) + b[:, :, :, i]
    return activation(result)
