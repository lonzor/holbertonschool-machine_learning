#!/usr/ben/env python3
"""
Contains function gensim_to_keras()
"""

import gensim.models
import tensorflow.keras


def gensim_to_keras(model):
    """
    Converts a gensim word2vac model to keras embedding model
    """
    embed = model.wv.get_keras_embedding(train_embeddings=True)
    return (embed)
