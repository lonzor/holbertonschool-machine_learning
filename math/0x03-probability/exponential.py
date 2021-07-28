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

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period
        x is the time period
        returns the PDF (probability density function) value for x
        if x is out of range, return 0
        """
        lambtha = self.lambtha
        e = 2.7182818285

        if x < 0:
            return 0
        prob_den = lambtha * (e ** (-lambtha * x))
        return prob_den

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given time period
        x is the time period
        Returns the CDF (cumulative dist) value for x
        If x is out of range, return 0
        """
        lambtha = self.lambtha
        e = 2.7182818285

        if x < 0:
            return 0
        cum_dist = 1 - (e ** (-lambtha * x))
        return cum_dist
