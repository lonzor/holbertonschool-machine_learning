#!/usr/bin/env python3
"""
trains a model using mini-batch gradient descent
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, verbose=True,
                shuffle=False):
    """
    network is the model to train
    data is a numpy.ndarray of shape (m, nx)
    - containing the input data.
    labels is a one-hot numpy.ndarray of shape (m, classes)
    - containing the labels of data.
    batch_size is the size of the batch used for
    - mini-batch gradient descent.
    epochs is the number of passes through data for
    - mini-batch gradient descent.
    verbose is a boolean that determines if output should
    - be printed during training.
    Returns history object
    """
    hist_obj = network.fit(x=data, y=labels, batch_size=batch_size,
                           epochs=epochs, verbose=verbose, shuffle=shuffle)
    return hist_obj
