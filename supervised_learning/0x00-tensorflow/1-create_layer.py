#!/usr/bin/env python3
"""Module hold create_layer method"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    prev is the tensor output of the previous layer
    n is the number of nodes in the layer to create
    activation is the activation function that the layer should use
    """
    weight = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, activation, kernel_initializer=weight,
                            name="layer")
    return layer(prev)
