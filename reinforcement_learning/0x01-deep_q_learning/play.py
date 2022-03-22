#!/usr/bin/env python3
"""
Displays game
"""
import gym
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
import keras as K

create_model = __import__('train').create_model
BreakoutProcessor = __import__('train').BreakoutProcessor

if __name__ = '__main__':
    env = gym.make("Breakout-v0")
    env.reset()
    act_count = env.action_space.n
    window = 4
    model = create_model(act_count, window)
    memory = SequentialMemory(limit=100000, window_length)
    processor = BreakoutProcessor()
    policy = GreedyQPolicy()
    dqn_agent = DQNAgent(model=model, nb_actions=act_count,
                         test_policy=policy, processor=processor,
                         memory=memory)
    dqn_agent.compile(K.optimizers.Adam(lr=.00025), metrics=['mae'])
    dqn_agent.load_weights('policy.h5')
    dqn_agent(env, nb_episodes=10, visualize=True)
