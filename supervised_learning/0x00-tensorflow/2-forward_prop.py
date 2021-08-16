#!/usr/bin/env python3
"""Contains method for forward prop using tensorflow"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    x is the placeholder for the input data
    layer_sizes is a list containing the number of nodes
    in each layer of the network
    activations is a list containing the activation
    functions for each layer of the network
    """
    result = x
    for i in range(len(layer_sizes)):
        result = create_layer(result, layer_sizes[i], activations[i])
    return result
