#!/user/bin/env python3
"""
Contains function load_frozen_lake()
"""
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    loads the pre-made FrozenLakeEnv evnironment from OpenAI’s gym
    """
    env = gym.make(id='FrozenLake-v1', desc=desc, map_name=map_name,
                   is_slippery=is_slippery)

    return env
