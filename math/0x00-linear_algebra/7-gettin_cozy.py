#!/usr/bin/env python3
"""concatenates two matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Returns a new concatenated matrix.
    If the two matrices cannot be concatenated return None.
    """
    concat_matrix = []
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        for r in mat1:
            concat_matrix.append(list(r))
        for r in mat2:
            concat_matrix.append(list(r))

    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        for r, i in zip(mat1, mat2):
            concat_matrix.append(list(r) + i)

    return concat_matrix
