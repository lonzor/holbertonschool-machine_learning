#!/usr/bin/env python3
"""module for Poisson distribution"""


class Poisson:
    """class modeled after the Poisson distribution"""
    def __init__(self, data=None, lambtha=1.):
        """initialize/constructor for class Poisson"""
        if data is None:
            self.lambtha = float(lambtha)
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = float(sum(data) / len(data))
