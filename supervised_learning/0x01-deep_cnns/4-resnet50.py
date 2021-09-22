#!/usr/bin/env python3
"""
contains function resnet50
"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    builds a resnet-50 architecture keras model
    """
    init = K.initializers.he_normal()
    input_data = K.Input(shape=(224, 224, 3))

    c_layer1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7),
                               padding='same', strides=(2, 2),
                               kernel_initializer=init)(input_data)
    b_norm1 = K.layers.BatchNormalization(axis=3)(c_layer1)
    activation1 = K.layers.Activation('relu')(b_norm1)
    pool = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                 padding='same')(activation1)

    filters = [64, 64, 256]
    p_block1 = projection_block(pool, filters, s=1)
    i_block_1a = identity_block(p_block1, filters)
    i_block_1b = identity_block(i_block_1a, filters)

    filters = [128, 128, 512]
    p_block2 = projection_block(i_block_1b, filters)
    i_block_2a = identity_block(p_block2, filters)
    i_block_2b = identity_block(i_block_2a, filters)
    i_block_2c = identity_block(i_block_2b, filters)

    filters = [256, 256, 1024]
    p_block3 = projection_block(i_block_2c, filters)
    i_block_3a = identity_block(p_block3, filters)
    i_block_3b = identity_block(i_block_3a, filters)
    i_block_3c = identity_block(i_block_3b, filters)
    i_block_3d = identity_block(i_block_3c, filters)
    i_block_3e = identity_block(i_block_3d, filters)

    filters = [512, 512, 2048]
    p_block4 = projection_block(i_block_3e, filters)
    i_block_4a = identity_block(p_block4, filters)
    i_block_4b = identity_block(i_block_4a, filters)

    pool2 = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1),
                                      padding='same')(i_block_4b)

    soft_max = K.layers.Dense(units=1000, activation='softmax',
                              kernel_initializer=init)(pool2)

    resnet_output = K.Model(inputs=input_data, outputs=soft_max)
    return resnet_output
