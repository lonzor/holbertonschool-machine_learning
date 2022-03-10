#!/usr/bin/env python3
"""
Contains function train()
Trains agent
"""


import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Starts q-learning
    """
    tot_rewards = []
    max_eps = epsilon

    for ep in range(episodes):
        fin = False
        epi_rewards = 0
        state = env.reset()

        for step in range(max_steps):
            act = epsilon_greedy(Q, state, epsilon)
            n_state, reward, fin, _ = env.step(act)

            if fin and reward == 0:
                reward = -1

            Q[state, act] = Q[state, act] + alpha *\
                (reward + gamma * np.max(Q[n_state, :]) - Q[state, act])

            state = n_state
            epi_rewards += reward

            if fin == True:
                break

        epsilon = min_epsilon + (max_eps - min_epsilon) * \
            np.exp(-epsilon_decay * ep)

        tot_rewards.append(epi_rewards)

    return Q, tot_rewards
