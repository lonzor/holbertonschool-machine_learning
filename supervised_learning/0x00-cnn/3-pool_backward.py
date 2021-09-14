#!/usr/bin/env python3
"""
Contains function pool_backward
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    performs back propagation over a pooling layer of a neural network
    """
    _, h, w, c = A_prev.shape
    m, da_h, da_w, da_c = dA.shape
    kern_h, kern_w = kernel_shape
    sh, sw = stride
    da = np.zeros_like(A_prev)

    for i in range(m):
        for j in range(da_h):
            a = sh * j
            for k in range(da_w):
                b = sw * k
                for l in range(da_c):
                    if mode == 'avg':
                        avg = dA[i, j, k, l] / kern_h / kern_w
                        da[i, a + kern_h, b: b + kern_w, l] += (
                            np.ones((kern_h, kern_w)) * avg
                        )
                    if mode == 'max':
                        sliced = A_prev[i, a: a + kern_h, b: b + kern_w, l]
                        mask = (sliced == np.max(sliced))
                        da[i, a: a + kern_h, b: b + kern_w, l] += (
                            mask * dA[i, j, k, l]
                        )
    return da
