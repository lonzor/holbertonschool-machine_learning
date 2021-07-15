#!/usr/bin/env python3
""" Calculates the shape of a matrix """



def matrix_shape(matrix):
    """returns list of ints as the shape of matrix"""
    shape_of_matrix = []
    if type(matrix) == list:
        shape_of_matrix.append(len(matrix))
        shape_of_matrix.extend(matrix_shape(matrix[0]))
    return shape_of_matrix
        
