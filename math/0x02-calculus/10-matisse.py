#!/usr/bin/env python3
"""calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """
    poly is a list of coeffecients representing a polynomial
    if poly is not valid, return None
    if the derivative is 0, then return [0]
    the derivative must be a list
    """
    if (type(poly) is not list or len(poly) < 1):
        return None

    der = [poly[ele] * ele for ele in range(len(poly))]
    if (sum(der) == 0):
        return [0]
    return der[1:]
