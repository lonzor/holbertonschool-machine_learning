#!/usr/bin/env python3
"""
contains function projection_block
"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    creates a project block
    """
    F11, F3, F12 = filters
    init = K.initializers.he_normal(seed=None)

    c_layer1 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1),
                               strides=(s, s), padding='same',
                               kernel_initializer=init)(A_prev)
    b_norm1 = K.layers.BatchNormalization(axis=3)(c_layer1)
    activation1 = K.layers.Activation('relu')(b_norm1)

    c_layer2 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3),
                               padding='same',
                               kernel_initializer=init)(activation1)
    b_norm2 = K.layers.BatchNormalization(axis=3)(c_layer2)
    activation2 = K.layers.Activation('relu')(b_norm2)

    c_layer3 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer=init)(activation2)
    b_norm3 = K.layers.BatchNormalization(axis=3)(c_layer3)

    c_layer1_F12 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                                   strides=(s, s), padding='same',
                                   kernel_initializer=init)(A_prev)
    b_norm4 = K.layers.BatchNormalization(axis=3)(c_layer1_F12)

    add = K.layers.Add()([b_norm3, b_norm4])
    projection_block = K.layers.Activation('relu')(add)
    return projection_block
