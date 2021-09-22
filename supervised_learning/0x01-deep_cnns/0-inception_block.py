#!/usr/bin/env python3
"""
Contains function inception_block
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    builds an inception block
    """
    init = K.initializers.he_normal(seed=None)
    F1, F3R, F3, F5R, F5, FPP = filters

    c_layer1 = K.layers.Conv2D(filters=F1, kernel_size=(1, 1),
                               padding='same',
                               activation='relu',
                               kernel_initializer=init)(A_prev)

    c_layer3R = K.layers.Conv2D(filters=F3R, kernel_size=(1, 1),
                                padding='same',
                                activation='relu',
                                kernel_initializer=init)(A_prev)

    c_layer3 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3),
                               padding='same',
                               activation='relu',
                               kernel_initializer=init)(c_layer3R)

    c_layer5R = K.layers.Conv2D(filters=F5R, kernel_size=(1, 1),
                                padding='same',
                                activation='relu',
                                kernel_initializer=init)(A_prev)

    c_layer5 = K.layers.Conv2D(filters=F5, kernel_size=(5, 5),
                               padding='same',
                               activation='relu',
                               kernel_initializer=init)(c_layer5R)

    c_pooling = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1),
                                      padding='same')(A_prev)

    c_layer_pool = K.layers.Conv2D(filters=FPP, kernel_size=(1, 1),
                                   padding='same', activation='relu',
                                   kernel_initializer=init)(c_pooling)

    concat_layer = K.layers.concatenate([c_layer1, c_layer3, c_layer5,
                                        c_layer_pool])
    return concat_layer
