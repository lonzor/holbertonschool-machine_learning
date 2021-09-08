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

    out_h = h - kern_h + 1
    out_w = w - kern_w + 1

    result = np.zeros((m, out_h, out_w))
    image = np.arange(m)
    for i in range(out_h):
        for j in range(out_w):
            result[image, i, j] = (np.sum(images[image, i: kern_h + i,
                                          j: kern_w + j] * kernel,
                                          axis=(1, 2)))
    return result
