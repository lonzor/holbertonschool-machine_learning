#!/usr/bin/env python3
"""
save_config saves a models cong file in JSON format
load_config loads a model that's in JSON format
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    network is the model whose configuration should be saved
    filename is the path of the file that the configuration should be saved to
    """
    config_file = network.to_json()
    with open(filename, 'w') as f:
        f.write(config_file)


def load_config(filename):
    """
    filename is the path of the file containing
    - the model’s configuration in JSON format.
    """
    with open(filename, 'r') as f:
        config_file = f.read()
    return K.models.model_from_json(config_file)
