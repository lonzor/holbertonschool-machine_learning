#!/usr/bin/env python3
"""
Contains method that performs a valid convolution on grayscale images
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    performs a valid convolution on grayscale images
    """
    h = images.shape[1]
    w = images.shape[2]

    kern_h = kernel.shape[0]
    kern_w = kernel.shape[1]
    m = images.shape[0]

    out_h = int(h - kern_h + 1)
    out_w = int(w - kern_w + 1)

    result = np.zeros((m, out_h, out_w))
    for i in range(out_w):
        for j in range(out_h):
            result[:, i, j] = (kernel * images[:, i: i + kern_h,
                                               j: j + kern_w])\
                                               .sum(axis=(1, 2))
    return result
