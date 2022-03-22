#!/usr/bin/env python3
"""
Trains agent in order to play Atari Breakout
"""

import numpy as np
import gym
from PIL import Image
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute, Input
from keras.optimizers import Adam
import keras as K
from rl.agents.dgn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


class BreakoutProcessor(Processor):
    """
    Class used for Atari Breakout Processor
    """

    def process_observation(self, obs):
        """
        Process observation method
        """
        assert obs.ndim == 3
        img = Image.fromarray(obs)
        img = img.resize(84, 84), Image.ANTIALIAS).convert('L')
        process_obs = np.array(img)
        assert process_obs.shape == (84, 84)
        return process_obs.astype('uint8')

    def process_state_batch(self, batch):
        """
        State batch method
        """
        process_batch = batch.astype('float32') / 255
        return process_batch

    def process_reward(self, reward):
        """
        process reward method
        """
        return np.clip(reward, -1., 1.)


def create_model(actions, window):
    """
    Creates model for deep q-learning
    """
    inputs = Input(shape=(window, 84, 84))
    model = Permute((2, 3, 1))(data_input)
    model = Conv2D(32, 8, strides=4, activation="relu",
                   data_format="channels_last")(model)
    model = Conv2D(64, 4, strides=2, activation="relu",
                   data_format="channels_last")(model)
    model = Conv2D(64, 3, strides=1, activation="relu",
                   data_format="channels_last")(model)
    model = Flatten()(model)
    model = Dense(512, activation="relu")(model)
    results = Dense(actions, activation="linear"(model)

    return K.Model(inputs=inputs, results=output)


if __name__ == '__main__':
    env = gym.make("Breakout-v0")
    env.seed(1)
    env.reset()
    nb_actions = env.action_space.n
    model = create_model(nb_actions, 4)
    mem = SequentialMemory(limit=100000, window_length=4)
    processor = BreakoutProcessor()
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                  value_max=1., value_min=.1,
                                  value_test=.05, nb_steps=1000)

    dqn_agent = DQNAgent(model=model, nb_actions=nb_actions, policy=policy,
                         memory=memory, processor=processor,
                         nb_steps_warmup=1000, gamma=.99,
                         target_model_update=100, train_interval=4,
                         delta_clip=1.)

    dqn_agent.compile(Adam(lr=.00025), metrics=['mae'])
    dqn_agent.fit(env, nb_steps=175000, log_interval=10000,
                  visualize=False, verbose=2)
    model.save_weights('policy.h5', overwrite=True)
