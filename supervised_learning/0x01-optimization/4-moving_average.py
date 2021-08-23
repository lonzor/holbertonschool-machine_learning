#!/usr/bin/env python3
"""calculates moving average"""


def moving_average(data, beta):
    """
    data is the list of data to calculate the moving average of
    beta is the weight used for the moving average
    Your moving average calculation should use bias correction
    Returns: a list containing the moving averages of data
    """
    m_avg = []
    vals = 0
    for i in range(len(data)):
        vals = beta * vals + (1 - beta) * data[i]
        m_avg.append(vals / (1 - beta ** (i + 1)))
    return m_avg
