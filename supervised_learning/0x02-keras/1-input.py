#!/usr/bin/env python3
"""
builds a neural network with the Keras library
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    nx is the number of input features to the network
    layers is a list containing the number of nodes
    - in each layer of the network
    activations is a list containing the activation functions
    - used for each layer of the network
    lambtha is the L2 regularization parameter
    keep_prob is the probability that a node will be kept for dropout
    Returns: the keras model
    """
    inputs = K.Input(shape=(nx,))
    regul = K.regularizers.l2(lambtha)
    la = K.layers.Dense(layers[0], activation=activations[0],
                        kernel_regularizer=regul)(inputs)
    for lay in range(1, len(layers)):
        la = K.layers.Dropout(1 - keep_prob)(la)
        la = K.layers.Dense(layers[lay], activation=activations[lay],
                            kernel_regularizer=regul)(la)
    keras_model = K.Model(inputs=inputs, outputs=la)
    return keras_model
