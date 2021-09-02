#!/usr/bin/env python3
"""
builds a neural network with the Keras library
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    nx is the number of input features to the network
    layers is a list containing the number of nodes
    - in each layer of the network.
    activations is a list containing the activation functions
    - used for each layer of the network.
    lambtha is the L2 regularization parameter
    keep_prob is the probability that a node will be kept for dropout
    returns the Keras model
    """
    layer_count = len(layers)
    model = K.Sequential()
    regul = K.regularizers.l2(lambtha)
    for lay in range(layer_count):
        model.add(K.layers.Dense(layers[lay], kernel_regularizer=regul,
                  activation=activations[lay], input_dim=nx))

        if lay < layer_count - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
