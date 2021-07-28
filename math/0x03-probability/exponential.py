#!/usr/bin/env python3
"""
Contains class Exponential that represents an exponential distribution
"""


class Exponential:
    """class for exponential dist"""

    def __init__(self, data=None, lambtha=1.):
        """
        data is list of data to be used to estimate the distribution
        lambtha is expected number of occurences in a given time frame
        save lambtha as a float
        check if data is None or not
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            else:
                self.lambtha = float(lambtha)

        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            elif len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                self.lambtha = float(len(data) / sum(data))
