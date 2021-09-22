#!/usr/bin/env python3
"""
contains function densenet121
"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    builds a model of DenseNet-121 architecture
    """
    input_data = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal()

    b_norm1 = K.layers.BatchNormalization(axis=3)(input_data)
    act1 = K.layers.Activation('relu')(b_norm1)
    c_layer1 = K.layers.Conv2D(filters=2*growth_rate, strides=(2, 2),
                               kernel_size=(7, 7), padding='same',
                               kernel_initializer=init)(act1)
    m_pool = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                   padding='same')(c_layer1)

    d_block1, filters = dense_block(m_pool, 64, growth_rate, 6)
    t_layer1, filters = transition_layer(d_block1, filters, compression)
    d_block2, filters = dense_block(t_layer1, filters, growth_rate, 12)
    t_layer2, filters = transition_layer(d_block2, filters, compression)
    d_block3, filters = dense_block(t_layer2, filters, growth_rate, 24)
    t_layer3, filters = transition_layer(d_block3, filters, compression)
    d_block4, filters = dense_block(t_layer3, filters, growth_rate, 16)

    a_pool = K.layers.AveragePooling2D(pool_size=(7, 7), padding='same',
                                       strides=(1, 1))(d_block4)

    soft_max = K.layers.Dense(units=1000, activation='softmax',
                              kernel_initializer=init)(a_pool)

    dense_net_model = K.Model(inputs=input_data, outputs=soft_max)
    return dense_net_model
