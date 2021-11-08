#!/usr/bin/env python3
"""
contains function determinant
"""


def determinant(matrix):
    """
    find the determinant of a matrix
    """
    if len(matrix[0]) == 0:
        return 1
    if len(matrix[0]) == 1:
        return matrix[0][0]
    if len(matrix[0]) == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    if matrix == []:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    matrix_list = list(range(len(matrix)))
    deter = 0

    for x in matrix_list:
        matrix_copy = matrix
        matrix_copy = matrix_copy[1:]
        h = len(matrix_copy)
        for i in range(h):
            matrix_copy[i] = matrix_copy[i][:x] + matrix_copy[i][x+1:]
    return deter
