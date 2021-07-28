#!/usr/bin/env python3
"""
Contains class Binomial
represents a binomial distribution
"""


class Binomial():
    """binomial dist"""

    def __init__(self, data=None, n=1, p=0.5):
        """
        data is a list of the data to be used to estimate the distribution
        n is the number of Bernoulli trials
        p is the probability of a “success”
        calculates n and p from data
        """
        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            elif p <= 0 or p >= 1:
                raise ValueError('p must be greater than 0 and less than 1')
            else:
                self.n = n
                self.p = p

        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            elif len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                mean = float(sum(data) / len(data))
                dev = [(i - mean) ** 2 for i in data]
                vari = sum(dev) / len(data)
                q = vari / mean
                p = 1 - q
                n = round(mean / p)
                p = float(mean / n)
                self.n = n
                self.p = p
