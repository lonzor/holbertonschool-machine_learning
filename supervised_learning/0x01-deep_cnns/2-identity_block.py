#!/usr/bin/env python3
"""
contains function identity_block
"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    builds identity block
    """
    F11, F3, F12 = filters
    init = K.initializers.he_normal(seed=None)

    c_layer1 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer=init)(A_prev)

    b_norm1 = K.layers.BatchNormalization(axis=3)(c_layer1)
    activ_b_norm1 = K.layers.Activation('relu')(b_norm1)
    c_layer2 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3),
                               padding='same',
                               kernel_initializer=init)(activ_b_norm1)
    b_norm2 = K.layers.BatchNormalization(axis=3)(c_layer2)
    activ_b_norm2 = K.layers.Activation('relu')(b_norm2)

    c_layer3 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer=init)(activ_b_norm2)
    b_norm3 = K.layers.BatchNormalization(axis=3)(c_layer3)

    add = K.layers.Add()([b_norm3, A_prev])

    i_block = K.layers.Activation('relu')(add)
    return i_block
