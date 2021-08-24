#!/usr/bin/env python3
"""
creates a batch normalization layer for a neural network in tensorflow
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    prev is the activated output of the previous layer
    n is the number of nodes in the layer to be created
    activation is the activation function that should be used
    - on the output of the layer
    """
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    base = tf.layers.dense(prev, n, kernel_initializer=w)
    m, var = tf.nn.moments(base, 0)
    beta = tf.Variable(tf.zeros(n), trainable=True)
    g = tf.Variable(tf.ones(n), trainable=True)
    epsilon = 1e-8
    batch_norm = tf.nn.batch_normalization(base, m, var, beta, g, epsilon)
    return activation(batch_norm)
