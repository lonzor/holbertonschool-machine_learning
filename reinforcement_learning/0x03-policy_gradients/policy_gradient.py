#!/usr/bin/env python3
"""
Contains policy() and policy_gradient() functions.
"""
import numpy as np


def policy(matrix, weight):
    """
    Finds policy and computes it.
    """
    x = np.dot(matrix, weight)
    ex = np.exp(x)
    result = ex / np.sum(exp)
    return result

def policy_gradient(state, weight):
    """
    Computes Monte-Carlo policy
    """
    actions = policy(state, weight)
    act = np.random.choice(len(actions[0]), p=actions([0])
    shape = actions.reshape(-1, 1)
    softmax np.diagflat(shape) - np.dot(shape, shape.T)
    softmax_d = softmax[act, :]
    log_d = softmax_d / actions[0, act]
    gradient = state.T.dot(log_d[None, :])

    return act, gradient
