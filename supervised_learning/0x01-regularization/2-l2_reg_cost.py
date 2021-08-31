#!/usr/bin/env python3
"""
calculates the cost of a neural network with L2 regularization
using TensorFlow
"""
import tensorflow as tf


def l2_reg_cost(cost):
    """
    cost is a tensor containing the cost of the network
    - without L2 regularization.
    Returns: a tensor containing the cost of the network
    - accounting for L2 regularization.
    """
    tf_reg_cost = cost + tf.losses.get_regularization_losses()
    return tf_reg_cost
