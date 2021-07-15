#!/usr/bin/env python3
"""Adds elements of two arrays"""


def add_arrays(arr1, arr2):
    """
    Arrays must be of the same shape.
    If not, then returns None.
    Adds values at the same position.
    """
    arr_sum = []

    if len(arr1) != len(arr2):
        return None

    for value in range(len(arr1)):
        arr_sum.append(arr1[value] + arr2[value])
    return arr_sum
