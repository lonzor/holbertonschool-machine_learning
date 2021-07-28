#!/usr/bin/env python3
"""
a class Normal that represents a normal distribution:
"""


class Normal:
    """class for normal distribution"""

    def __init__(self, data=None, mean=0, stddev=1.):
        """
        initializes and constructs
        calculates the mean and standard deviation of data
        """
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            else:
                self.stddev = float(stddev)
                self.mean = float(mean)

        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            elif len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                self.mean = float(sum(data) / len(data))
                d_var = sum([((n - self.mean)) ** 2 for n in data])/len(data)
                self.stddev = d_var ** 0.5

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value
        x is the x-value
        Returns the z-score of x
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score
        z is the z-score
        Returns the x-value of z
        """
        return (z * self.stddev) + self.mean
