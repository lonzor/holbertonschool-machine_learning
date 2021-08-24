#!/usr/bin/env python3
"""
creates a batch normalization layer for a neural network in tensorflow
"""
import tensorflow as tf


def createf_batch_norm_layer(prev, n, activation):
    """
    prev is the activated output of the previous layer
    n is the number of nodes in the layer to be created
    activation is the activation function that should be used
    - on the output of the layer
    """
    kern = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    base = tf.layers.dense(prev, n, kernel_initializer=kern)
    m, var = tf.nn.moments(base(prev), axes=[0])
    g = tf.Variable(tf.ones(n), trainable=True)
    beta = tf.Variable(tf.zeros(n), trainable=True)
    epsilon = 1e-8

    batch = tf.nn.batch_normalization(base, m, var, beta, g, epsilon)
    return activation(batch)
