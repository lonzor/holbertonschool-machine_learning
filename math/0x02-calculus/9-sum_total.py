#!/usr/bin/env python3
"""function that calculates a summation"""


def summation_i_squared(n):
    """
    n is the stopping condition
    if n is not valid, return None
    loops are not allowed
    return the int value of the sum
    """
    if type(n) is not int or n < 1:
        return None

    summation = (n * (n + 1) * ((n * 2) + 1)) / 6
    return int(summation)
