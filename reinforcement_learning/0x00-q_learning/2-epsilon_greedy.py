#!/usr/bin/env python3
"""
Contains function epsilon_greedy()
"""
import numpy as np
import gym


def epsilon_greedy(Q, state, epsilon):
    """
    Function used to determine next action.
    """
    if np.random.uniform() < epsilon:
        return np.random.randint(0, Q.shape[1])
    else:
        return np.argmax(Q[state, :])
