#!/usr/bin/env python3
"""
Initializes Q-Table
"""
import numpy as np
import gym


def q_init(env):
    """
    env: instance of FrozenLakeEnv
    Initializer to create a Q-Table
    """
    a_space_size = env.action_space.n
    s_space_size = env.observation_space.n

    q_table = np.zeros((s_space_size, a_space_size))

    return q_table
