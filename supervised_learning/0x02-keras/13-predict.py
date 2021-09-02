#!/usr/bin/env python3
"""
contains function that makes a prediction using a neural network
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    network is the network model to make the prediction with
    data is the input data to make the prediction with
    verbose is a boolean that determines if output should be printed
    """
    prediction = network.predict(data, verbose=verbose)
    return prediction
