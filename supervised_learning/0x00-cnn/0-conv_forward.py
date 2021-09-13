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
    kern_h, kern_w, cn = W.shape[0], W.shape[1], W.shape[2]
    sh, sw = stride
    if padding == 'same':
        pad_h = int(((h - 1) * sh + kern_h - h) / 2)
        pad_w = int(((w - 1) * sw + kern_w - w) / 2)

    else:
        pad_h = 0
        pad_w = 0

    next_h = int(((h - kern_h + (2 * pad_h)) / sh + 1))
    next_w = int(((w - kern_w + (2 * pad_w)) / sw + 1))
    result = np.zeros((m, next_h, next_w, cn))
    pad_s = ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0))
    pad_img = np.pad(A_prev, pad_width=pad_s, mode='constant',
                    constant_values=0)

    for i in range(next_h):
        x = i * sh
        for j in range(next_w):
            y = j * sw
            for k in range(cn):
                img = pad_img[:, x:x + kern_h, y:y + kern_w, :]
            kernel = W[:, :, :, k]
            result[:, i, j, k] = np.sum(np.multiply(img, kernel),
                                        axis=(1, 2, 3))

    result = result + b
    return activation(result)
