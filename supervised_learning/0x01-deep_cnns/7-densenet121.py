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

    X_output = K.layers.BatchNormalization(axis=3)(input_data)
    X_output = K.layers.Activation('relu')(X_output)
    X_output = K.layers.Conv2D(filters=2*growth_rate, strides=(2, 2),
                               kernel_size=(7, 7), padding='same',
                               kernel_initializer=init)(X_output)
    X_output = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                     padding='same')(X_output)

    X_output, filters = dense_block(X_output, 64, growth_rate, 6)
    X_output, filters = transition_layer(X_output, filters, compression)
    X_output, filters = dense_block(X_output, filters, growth_rate, 12)
    X_output, filters = transition_layer(X_output, filters, compression)
    X_output, filters = dense_block(X_output, filters, growth_rate, 24)
    X_output, filters = transition_layer(X_output, filters, compression)
    X_output, filters = dense_block(X_output, filters, growth_rate, 16)

    X_output = K.layers.AveragePooling2D(pool_size=(7, 7), padding='same',
                                         strides=(1, 1))(X_output)

    X_output = K.layers.Dense(units=1000, activation='softmax',
                              kernel_initializer=init)(X_output)

    dense_net_model = K.Model(inputs=input_data, outputs=X_output)
    return dense_net_model
