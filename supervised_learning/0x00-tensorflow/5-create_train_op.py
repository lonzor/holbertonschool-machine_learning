#!/usr/bin/env python3
"""Contains method that creates training operation."""
import tensorflow as tf


def create_train_op(loss, alpha):
    """
    loss is the loss of the network’s prediction
    alpha is the learning rate
    """
    opt = tf.train.GradientDescentOptimizer(alpha)
    result = opt.minimize(loss)
    return result
