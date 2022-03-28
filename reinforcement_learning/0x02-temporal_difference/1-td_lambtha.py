#!/usr/bin/env python3
"""
Contains function td_lambtha()
"""
import gym
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Performs the td lambtha algorithm
    """
    prev_s = env.observation_space.n
    elig = np.zeros(prev_s)
    for _ in range(episodes):
        state = env.reset()
        for _ in range(max_steps):
            elig *= lambtha * gamma
            elig[state] += 1.0
            x = policy(state)
            n_state, reward, finished, _ = env.step(x)
            if env.desc.reshape(prev_s)[n_state] == b'G':
                reward = 1.0
            if env.desc.reshape(prev_s)[n_state] == b'H':
                reward = -1.0
            delt = reward + gamma * V[n_state] - V[state]
            V[state] = V[state] + alpha * delt * elig[state]
            if done:
                break
            state = n_state
    return V
