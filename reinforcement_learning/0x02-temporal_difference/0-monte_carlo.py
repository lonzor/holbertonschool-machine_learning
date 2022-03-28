#!/usr/bin/env python3
"""
Contains function monte_carlo()
"""
import gym
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99)
    """
    Function performs Monte Carlop
    """
    space = env.observation_space.n
    disc = [gamma**i for i in range(max_steps)]
    for eps in range(episodes):
        state = env.reset()
        eps = [[], []]
        for step in range(max_steps):
            act = policy(state)
            n_state, reward, _, _ = env.step(act)
            eps[0].append(state)
            if env.desc.reshape(space)[n_state] == b'G':
                eps[1].append(1)
                break
            if env.desc.reshape(space)[n_state] == b'H':
                eps[1].append(-1)
                break
            eps[1].append(reward)
            state = n_state
        for i in range(len(eps[0])):
            total = sum(np.array(eps[1][i:]) *
                        np.array(disc[:len(eps[1][i:])]))
            V[eps[0][i]] = V[eps[0][i]] +\
                alpha * (total - V[eps[0][i]])
    return V
