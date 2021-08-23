#!/usr/bin/env python3
"""
updates a variable using the RMSProp optimization algorithm
"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    alpha is the learning rate
    beta2 is the RMSProp weight
    epsilon is a small number to avoid division by zero
    var is a numpy.ndarray containing the variable to be updated
    grad is a numpy.ndarray containing the gradient of var
    s is the previous second moment of var
    Returns: the updated variable and the new moment, respectively
    """
    dw = beta2 * s + (1 - beta2) * (grad ** 2)
    new_var = var - alpha * (grad / (dw ** (1 / 2) + epsilon))
    return new_var, dw
