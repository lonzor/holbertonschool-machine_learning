#!/usr/bin/env python3
"""
contains function dense_block
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    builds a dense block
    """
    init = K.initializers.he_normal()

    for layer in range(layers):
        b_norm1 = K.layers.BatchNormalization(axis=3)(X)
        act1 = K.layers.Activation('relu')(b_norm1)
        c_layer1 = K.layers.Conv2D(filters=4 * growth_rate,
                                   kernel_size=(1, 1), padding='same',
                                   strides=(1, 1),
                                   kernel_initializer=init)(act1)

        b_norm2 = K.layers.BatchNormalization(axis=3)(c_layer1)
        act2 = K.layers.Activation('relu')(b_norm2)
        c_layer2 = K.layers.Conv2D(filters=growth_rate,
                                   kernel_size=(3, 3), padding='same',
                                   strides=(1, 1),
                                   kernel_initializer=init)(act2)

        X = K.layers.concatenate([X, c_layer2])
        nb_filters += growth_rate

    return X, nb_filters
