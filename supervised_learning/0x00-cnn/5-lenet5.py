#!/usr/bin/env python3
"""
Contains function lenet5
"""
import tensorflow.keras as K


def lenet5(X):
    """
    builds a modified version of the LeNet-5 architecture using keras
    """
    init = K.initializers.he_normal()
    con_1 = K.layers.Conv2D(filters=6, kernel_size=5, padding='same',
                            activation='relu', kernel_initializer=init)(X)
    pool_1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(con_1)
    con_2 = K.layers.Conv2D(filters=16, kernel_size=5, padding='valid',
                            activation='relu',
                            kernel_initializer=init)(pool_1)
    pool_2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(con_2)
    flat = K.layers.Flatten()(pool_2)

    c_layer1 = K.layers.Dense(units=120, activation='relu',
                              kernel_initializer=init)(flat)
    c_layer2 = K.layers.Dense(units=84, activation='relu',
                              kernel_initializer=init)(c_layer1)
    c_layer3 = K.layers.Dense(units=10, activation='softmax',
                              kernel_initializer=init)(c_layer2)

    create_model = K.models.Model(X, c_layer3)
    adam_train = K.optimizers.Adam()
    create_model.compile(optimizer=adam_train,
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])
    return create_model
