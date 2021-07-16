#!/usr/bin/env python3
"""performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """
    Must return a new matrix with the product of two matrices.
    If the two matrices cannot be mulitplied, return None.
    """
    if (len(mat1[0]) != len(mat2)):
        return None

    mult_matrix = []
    for i in range(len(mat1)):
        current_list = []
        for j in range(len(mat2[0])):
            product = 0
            for k in range(len(mat1[0])):
                product = product + (mat1[i][k] * mat2[k][j])
            current_list.append(product)
        mult_matrix.append(current_list)
    return mult_matrix
