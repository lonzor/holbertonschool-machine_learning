#!/usr/bin/env python3
"""
contains functions determinant, minor, cofactor, and adjugate
"""


def inverse(matrix):
    """
    finds the inverse of a matrix
    """
    adj_mat = adjugate(matrix)
    deter = determinant(matrix)

    if deter == 0:
        return None

    for row in range(len(adj_mat)):
        for col in range(len(adj_mat)):
            adj_mat[row][col] = adj_mat[row][col] * (1 / deter)

    return adj_mat


def adjugate(matrix):
    """
    finds the adjugate of a matrix
    """
    adjugate = []
    cofact = cofactor(matrix)

    for row in range(len(matrix)):
        adjugate.append([])
        for col in range(len(matrix)):
            adjugate[row].append(cofact[col][row])
    return adjugate


def cofactor(matrix):
    """
    finds the cofactor of a matrix
    """
    cofactor = minor(matrix)
    for row in range(len(cofactor)):
        for col in range(len(cofactor[0])):
            cofactor[row][col] *= ((-1) ** (row + col))
    return cofactor


def determinant(matrix):
    """
    find the determinant of a matrix
    """
    if type(matrix) is not list or not matrix:
        raise TypeError("matrix must be a list of lists")
    for r in matrix:
        if type(r) is not list:
            raise TypeError("matrix must be a list of lists")
    if len(matrix) > 0 and len(matrix[0]) > 0:
        if len(matrix) != len(matrix[0]):
            raise ValueError("matrix must be a square matrix")

    if len(matrix[0]) == 0:
        return 1
    if matrix == ([]):
        return 1
    if len(matrix[0]) == 1:
        return matrix[0][0]
    if len(matrix[0]) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    coeff = 1
    deter = 0

    for j in range(len(matrix[0])):
        matrix_cpy = matrix[1:]
        for i in range(len(matrix_cpy)):
            matrix_cpy[i] = matrix_cpy[i][0:j] + matrix_cpy[i][j + 1:]
        deter += (coeff * matrix[0][j]) * determinant(matrix_cpy)
        coeff = coeff * -1
    return deter


def minor(matrix):
    """
    finds the minor of a matrix
    """
    for m in matrix:
        if type(m) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(matrix) != len(m):
            raise ValueError("matrix must be a non-empty square matrix")
    if type(matrix) is not list or not matrix:
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1:
        return [[1]]

    minor = []
    for i in range(len(matrix)):
        sub = []
        for j in range(len(matrix)):
            copied_matrix = []
            for idx in matrix:
                copied_matrix.append(idx.copy())
            del copied_matrix[i]
            for k in copied_matrix:
                del k[j]
            sub.append(determinant(copied_matrix))
        minor.append(sub)
    return minor
