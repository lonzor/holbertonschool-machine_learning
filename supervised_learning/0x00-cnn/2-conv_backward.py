#!/usr/bin/env python3
"""
contains function conv_backward
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding='same', stride=(1, 1)):
    """
    performs back propagation over a convolutional layer of a neural network
    """
    _, h, w, c = A_prev.shape
    m, dZ_h, dZ_w, dZ_c = dZ.shape
    kern_h, kern_w, _, _ = W.shape
    sh, sw = stride
    dw = np.zeros_like(W)
    da = np.zeros_like(A_prev)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    if padding == 'same':
        ph = (((h - 1) * sh) + kern_h - h) // 2 + 1
        pw = (((w - 1) * sw) + kern_w - w) // 2 + 1
    if padding == 'valid':
        ph = 0
        pw = 0
    pad_img = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                     mode='constant', constant_values=0)
    pad_img2 = np.pad(da, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                      mode='constant', constant_values=0)

    for i in range(m):
        for j in range(dZ_h):
            for k in range(dZ_w):
                for l in range(dZ_c):
                    dw[:, :, :, l] += (np.multiply(
                        pad_img[i, sh*j: sh*j+kern_h, sw*k: sw*k+kern_w, :],
                        dZ[i, j, k, l]
                    ))
                    pad_img2[i, sh*j: sh*j+kern_h, sw*k: sw*k+kern_w, :] += (
                        np.multiply(W[:, :, :, l], dZ[i, j, k, l])
                    )
    if padding == 'same':
        da = pad_img2[:, ph: -ph, pw: -pw, :]
    else:
        da = pad_img2
    return da, dw, db
