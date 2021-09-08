#!/usr/bin/env python3
"""
Contains function pool
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Function that performs pooling on images
    """
    m, h = images.shape[0], images.shape[1]
    w, c = images.shape[2], images.shape[3]
    kern_h, kern_w = kernel.shape[0], kernel.shape[1]
    sh, sw = stride[0], stride[1]
    out_h = int(((h - kern_h) / sh) + 1)
    out_w = int(((w - kern_w) / sw) + 1)
    result = np.zeros((m, out_h, out_w, c))

    for i in range(out_h):
        x = i * sh
        for j in range(out_w):
            y = j * sw
            img = images[:, x:x + kern_h, y:y + kern_w, :]
            if mode == 'max':
                result[:, i, j, :] = np.max(img, axis=(1, 2))
            else:
                result[:, i, j, :] = np.average(img, axis=(1, 2))

    return result
