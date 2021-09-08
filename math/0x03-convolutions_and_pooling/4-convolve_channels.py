#!/usr/bin/env python3
"""
contains function convolve_channels
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    performs a convolution on images with channels
    performs a convolution on images with channels:
    """
    m, h = images.shape[0], images.shape[1]
    w, c = images.shape[2], images.shape[3]
    kern_h, kern_w = kernel.shape[0], kernel.shape[1]
    st_h, st_w = stride[0], stride[1]

    if type(padding) is tuple:
        pad_h = padding[0]
        pad_w = padding[1]
    elif padding == 'same':
        pad_h = int(((h - 1) * st_h + kern_h - h) / 2) + 1
        pad_w = int(((w - 1) * st_w + kern_w - w) / 2) + 1
    else:
        pad_h, pad_w = 0, 0

    pad_s = ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0))
    pad_im = np.pad(images, pad_width=pad_s, mode='constant',
                    constant_values=0)

    out_h = int((pad_im.shape[1] - kern_h) / st_h + 1)
    out_w = int((pad_im.shape[2] - kern_w) / st_w + 1)
    result = np.zeros((m, out_h, out_w))

    for i in range(out_w):
        for j in range(out_h):
            result[:, j, i] = (kernel * pad_im[:,
                                               j * st_h: j * st_h + kern_h,
                                               i * st_w: i * st_w + kern_w,
                                               :]).sum(axis=(1, 2, 3))
    return result
