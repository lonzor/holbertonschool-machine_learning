#!/usr/bin/env python3
"""
contains function convolve_grayscale
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    performs a convolution on grayscale images with stride
    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kern_h, kern_w = kernel.shape[0], kernel.shape[1]
    st_h, st_w = stride[0], stride[1]

    if type(padding) is tuple:
        pad_h, pad_w = padding[0], padding[1]
    if padding == 'same':
        pad_h = int(((h - 1) * st_h + kern_h - h) / 2) + 1
        pad_w = int(((w - 1) * st_w + kern_w - w) / 2) + 1
    else:
        pad_h = 0
        pad_w = 0

    pad_s = ((0, 0), (pad_h, pad_h), (pad_w, pad_w))
    pad_im = np.pad(images, pad_width=pad_s, mode='constant',
                    constant_values=0)

    out_h = int((pad_im.shape[1] - kern_h) / st_h + 1)
    out_w = int((pad_im.shape[2] - kern_w) / st_w + 1)
    result = np.zeros((m, out_h, out_w))

    for i in range(out_w):
        for j in range(out_h):
            result[:, j, i] = (kernel * pad_im[:,
                                               j * st_h: j * st_h + kern_h,
                                               i * st_w: i * st_w + kern_w])\
                                               .sum(axis=(1, 2))
    return result
