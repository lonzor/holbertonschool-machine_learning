#!/usr/bin/env python3
"""
Contains function that generates variational autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Generates a variational autoencoder
    """
    inputs = keras.Input(shape=(input_dims,))
    encoded = keras.layers.Dense(hidden_layers[0], activation='relu')(inputs)

    for i in hidden_layers[1:]:
        encoded = keras.layers.Dense(i, activation='relu')(encoded)

    z_mean = keras.layers.Dense(latent_dims)(encoded)
    z_log = keras.layers.Dense(latent_dims)(encoded)

    def samples(args):
        """
        used for sampling
        """
        z_mean, z_log = args
        eps = keras.backend.random_normal(
            shape=(keras.backend.shape(z_mean)[0], latent_dims),
            mean=0, stddev=1)
        return z_mean + keras.backend.exp(z_log / 2) * eps

    z = keras.layers.Lambda(samples)([z_mean, z_log])
    encoder = keras.Model(inputs, [z_mean, z_log, z])

    inputs_latent = keras.Input(shape=(latent_dims,))
    decoded = keras.layers.Dense(hidden_layers[-1],
                                 activation='relu')(inputs_latent)
    for i in hidden_layers[-2::-1]:
        decoded = keras.layers.Dense(i, activation='relu')(decoded)

    decoded_out = keras.layers.Dense(input_dims,
                                     activation='sigmoid')(decoded)
    decoder = keras.Model(inputs_latent, decoded_out)
    outputs = decoder(encoder(inputs)[2])
    auto = keras.Model(inputs, outputs)

    def loss(inputs, outputs):
        """
        calculates loss
        """
        x_loss = keras.losses.binary_crossentropy(inputs, outputs)
        x_loss = x_loss * input_dims
        k1_loss = 1 + z_log - keras.backend.square(z_mean) \
            - keras.backend.exp(z_log)
        k1_loss = keras.backend.sum(k1_loss, axis=-1)
        k1_loss = k1_loss * 0.5
        v_loss = keras.backend.mean(x_loss + k1_loss)
        return v_loss

    auto.compile(optimizer='adam', loss=loss)

    return encoder, decoder, auto
