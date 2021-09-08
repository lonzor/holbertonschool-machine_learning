#!/usr/bin/env python3
"""
Contains function that performs convolution.
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    performs a same convolution on grayscale images
    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kern_h, kern_w = kernal.shape[0], kernel.shape[1]
    pad_h = kern_h // 2
    pad_w = kern_w // 2
    pad_s = ((0, 0), (pad_h, pad_h), (pad_w, pad_w))
    pad_im = np.pad(images, pad_width=pad_s, mode='constant',
                    constant_values=0)
    result = np.zeros((m, h, w))

    for i in range(w):
        for j in range(h):
            result[:, j, i] = (kernel * pad_im[:, j: j + kern_h,
                                               i: i + kern_w])\
                                               .sum(axis=(1, 2))
    return result
