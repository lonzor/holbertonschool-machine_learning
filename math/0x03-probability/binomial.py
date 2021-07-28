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

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes
        k is the number of successes
        Returns the PMF value for k
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0

        p = self.p
        n = self.n

        n_fact = 1
        for i in range(n):
            n_fact *= (i + 1)

        k_fact = 1
        for i in range(k):
            k_fact *= (i + 1)

        nk_fact = 1
        for i in range(n - k):
            nk_fact *= (i + 1)

        bi_co = n_fact / (k_fact * nk_fact)
        prob_mass = bi_co * (p ** k) * ((1 - p) ** (n - k))
        return prob_mass

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of successes
        k is the number of successes
        Returns the CDF value for k
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0

        cum_dist = 0
        for i in range(k + 1):
            cum_dist += self.pmf(i)
        return cum_dist
