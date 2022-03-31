#!/usr/bin/env python3
"""
Contains the following functions:
sarsa_lambtha()
epsilon_greedy()
"""
import gym
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs SARSA
    """
    def epsilon_greedy(state, Q, epsilon):
        """
        Greedy method
        """
        x = np.random.uniform(0, 1)
        if x >= epsilon:
            act = np.argmax(Q[state])
        else:
            act = np.random.randint(0, int(Q.shape[1]))
        return act

    state = env.observation_space.n
    init_ep = epsilon
    Et = np.zeroes((Q.shape))
    for eps in range(episodes):
        st = env.reset()
        action = epsilon_greedy(s, Q, epsilon=epsilon)
        for _ in range(max_steps):
            Et = Et * lambtha * gammaa
            Et[st, action] += 1.0
            n_state, reward, done, _ = env.step(action)
            n_act = epsilon_greedy(n_state, Q, epsilon=epsilon)
            if env.desc.reshape(state)[n_state] == b'H':
                reward = -1.0
            if env.desc.reshape(state)[n_state] == b'G':
                reward = -1.0
            delt = reward + gamma * Q[n_state, n_act] - Q[st, action]
            Q[st, action] = Q[st, action] + alpha * delt * Et[st, action]
            if done:
                break
            st = n_state
            action = n_act
        epsilon = min_epsilon + (init_ep - min_epsilon) *\
            np.exp(-epsilon_decay * eps)
    return Q
