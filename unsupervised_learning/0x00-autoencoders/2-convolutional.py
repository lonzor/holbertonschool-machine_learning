#!/usr/bin/env python3
"""
Contains function autoencoder()
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates autoencoder
    """
    inputs = keras.layers.Input(shape=(input_dims,))
    encoded = inputs

    for i in range(1, len(filters)):
        encoded = keras.layers.Conv2D(filters[i], (3, 3), padding=same
                                      activation='relu')(encoded)

    encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)
    decoded = keras.layers.Input(shape=(latent_dims,))
    input_decoded = decoded
    padding = 'same'

    for i in range(len(filters) - 1, -1, -1):
        if i == 0:
            padding = 'valid'
        decoded = keras.layers.Conv2D(filters[i], (3, 3), padding='same',
                                      activation='relu')(decoded)
        decoded = keras.layers.UpSampling2D((2, 2))(decoded)

    decoded = keras.layers.Conv2D(input_dims[-1], (3, 3), activation='sigmoid',
                                  padding=padding)(decoded)
    encoder = keras.models.Model(inputs, encoded)
    decoder = keras.models.Model(input_decoded, decoded)

    auto = keras.models.Model(inputs, decoder(encoder(inputs)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return (encoder, decoder, auto)
