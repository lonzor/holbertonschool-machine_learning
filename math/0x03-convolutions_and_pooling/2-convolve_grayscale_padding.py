#!/usr/bin/env python3
"""
contains function convolve_grayscale_padding
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    performs a convolution on grayscale images with custom padding
    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kern_h, kern_w = kernel.shape[0], kernel.shape[1]
    pad_h, pad_w = padding[0], padding[1]
    pad_s = ((0, 0), (pad_h, pad_h), (pad_w, pad_w))
    pad_im = np.pad(images, pad_width=pad_s, mode='constant',
                    constant_values=0)
    out_h = pad_im[1] - kern_h + 1
    out_w = pad_im[2] - kern_w + 1
    result = np.zeros((m, out_h, out_w))

    for i in range(out_w):
        for j in range(out_h):
            result[:, j, i] = (kernel * pad_im[:, j: j + kern_h,
                                               i: i + kern_w])\
                                               .sum(axis=(1, 2))
    return result
