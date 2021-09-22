#!/usr/bin/env python3
"""
contains function transition_layer
"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    creates a transition layer and returns filter count
    """
    init = K.initializers.he_normal()
    filter_num = int(nb_filters * compression)

    b_norm1 = K.layers.BatchNormalization(axis=3)(X)
    act1 = K.layers.Activation('relu')(b_norm1)
    c_layer1 = K.layers.Conv2D(filters=filter_num, padding='same',
                               kernel_size=(1, 1), strides=(1, 1),
                               kernel_initializer=init)(act1)
    pool = K.layers.AveragePooling2D(pool_size=(2, 2), padding='same',
                                     strides=(2, 2))(c_layer1)
    return pool, filter_num
