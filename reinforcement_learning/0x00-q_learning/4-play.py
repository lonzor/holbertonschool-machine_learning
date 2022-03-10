#!/usr/bin/env python3
"""
Contains function play()
Trained agent plays an episode
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    trained agent play an episode
    """
    state = env.reset()
    env.render()

    for step in range(max_steps):
        act = np.argmax(Q[state])
        n_state, reward, fin, _ = env.step(act)
        env.render()

        if fin:
            return reward
        state = n_state

    env.close()
