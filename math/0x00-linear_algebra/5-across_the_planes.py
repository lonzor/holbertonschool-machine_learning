#!/usr/bin/env python3
"""Adds the elements of two matrices at the same position"""


def add_matrices2D(mat1, mat2):
    """
    If mat1 and mat2 are not the same shape, returns None.
    A new matrix is created with the elements' addition.
    """
    mat_sum = []

    if len(mat1) != len(mat2):
        return None
    if len(mat1[0]) != len(mat2[0]):
        return None

    for r in range(len(mat1)):
        row = []
        for c in range(len(mat1[0])):
            row.append(mat1[r][c] + mat2[r][c])
        mat_sum.append(row)
    return mat_sum
