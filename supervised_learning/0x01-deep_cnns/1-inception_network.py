#!/usr/bin/env python3
"""
contains function inception_network
"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    builds an inception network based on 0-inception_block
    """
    init = K.initializers.he_normal(seed=None)
    input_data = K.Input(shape=(224, 224, 3))

    c_layer1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7),
                               strides=(2, 2), padding='same',
                               activation='relu',
                               kernel_initializer=init)(input_data)

    pool_layer1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                        padding='same')(c_layer1)

    c_layer2 = K.layers.Conv2D(filters=64, kernel_size=(1, 1),
                               padding='same', activation='relu',
                               kernel_initializer=init)(pool_layer1)

    c_layer3 = K.layers.Conv2D(filters=192, kernel_size=(3, 3),
                               padding='same', activation='relu',
                               kernel_initializer=init)(c_layer2)

    pool_layer3 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                        padding='same')(c_layer3)

    i_block1 = inception_block(pool_layer3, [64, 96, 128, 16, 32, 32])
    i_block2 = inception_block(i_block1, [128, 128, 192, 32, 96, 64])

    pool_block1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                        padding='same')(i_block2)

    i_block3 = inception_block(pool_block1, [192, 96, 208, 16, 48, 64])
    i_block4 = inception_block(i_block3, [160, 112, 224, 24, 64, 64])
    i_block5 = inception_block(i_block4, [128, 128, 256, 24, 64, 64])
    i_block6 = inception_block(i_block5, [112, 144, 288, 32, 64, 64])
    i_block7 = inception_block(i_block6, [256, 160, 320, 32, 128, 128])

    pool_block7 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                        padding='same')(i_block7)

    i_block8 = inception_block(pool_block7, [256, 160, 320, 32, 128, 128])
    i_block9 = inception_block(i_block8, [384, 192, 384, 48, 128, 128])

    avg_pool_block9 = K.layers.AveragePooling2D(pool_size=(7, 7),
                                                strides=(7, 7),
                                                padding='same')(i_block9)

    drop_avg_pool = K.layers.Dropout(0.4)(avg_pool_block9)

    soft_max = K.layers.Dense(1000, activation='softmax',
                              kernel_initializer=init)(drop_avg_pool)

    network = K.Model(inputs=input_data, outputs=soft_max)
    return network
