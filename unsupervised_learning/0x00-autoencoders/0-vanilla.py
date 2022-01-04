#!/usr/bin/env python3
"""
Contains function autoencoder()
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates autoencoder
    """
    inputs = keras.layers.Input(shape=(input_dims,))
    encoded = keras.layers.Dense(hidden_layers[0],
                                 activation='relu')(inputs)

    for i in range(1, len(hidden_layers)):
        encoded = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(encoded)

    encoded = keras.layers.Dense(latent_dims, activation='relu')(encoded)
    decoded = keras.layers.Input(shape=(latent_dims,))
    input_decoded = decoded

    for i in range(len(hidden_layers) - 1, -1, -1):
        decoded = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(decoded)

    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    encoder = keras.models.Model(inputs, encoded)
    decoder = keras.models.Model(input_decoded, decoded)

    auto = keras.models.Model(inputs, decoder(encoder(inputs)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return (encoder, decoder, auto)
