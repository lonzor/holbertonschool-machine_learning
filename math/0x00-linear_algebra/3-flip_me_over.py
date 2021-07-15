#!/usr/bin/env python3
"""Returns the transposed version of a 2D matrix"""


def matrix_transpose(matrix):
    """Returns a new matrix that was transposed"""
    trans_matrix = []
    for r in range(len(matrix[0])):
        trans_matrix.append([])
        for c in range(len(matrix)):
            trans_matrix[r].append(matrix[c][r])
    return trans_matrix
