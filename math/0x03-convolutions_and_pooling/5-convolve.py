#!/usr/bin/env python3
"""
contains the function convolve
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    performs a convolution on images using multiple kernels
    """
    m, h = images.shape[0], images.shape[1]
    w, c = images.shape[2], images.shape[3]
    kern_h, kern_w = kernels.shape[0], kernels.shape[1]
    nc = kernels.shape[3]
    sh, sw = stride[0], stride[1]

    if type(padding) is tuple:
        pad_h = padding[0]
        pad_w = padding[1]
    elif padding == 'same':
        pad_h = int(((h - 1) * sh + kern_h - h) / 2) + 1
        pad_w = int(((w - 1) * sw + kern_w - w) / 2) + 1
    else:
        pad_h, pad_w = 0, 0

    pad_s = ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0))
    pad_im = np.pad(images, pad_width=pad_s, mode='constant',
                    constant_values=0)
    out_h = int((pad_im.shape[1] - kern_h) / sh + 1)
    out_w = int((pad_im.shape[2] - kern_w) / sw + 1)
    result = np.zeros((m, out_h, out_w, nc))
    for i in range(out_h):
        x = i * sh
        for j in range(out_w):
            y = j * sw
            for k in range(nc):
                img = pad_im[:, x:x + kern_h, y:y + kern_w, :]
                kern = kernels[:, :, :, k]
                result[:, i, j, k] = np.sum(np.multiply(img, kern),
                                            axis=(1, 2, 3))
    return result
