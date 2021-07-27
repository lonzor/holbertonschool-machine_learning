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

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes
        k is the number of successes
        If k is not an integer, convert it to an integer
        If k is out of range, return 0
        returns prob_mass = PMF value for k
        """
        e = 2.7182818285
        fact = 1
        lamb = self.lambtha

        for i in range(k):
            fact = fact * (i + 1)
        prob_mass = ((lamb ** k) * (e ** -lamb)) / fact
        return prob_mass
