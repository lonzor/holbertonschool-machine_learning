#!/usr/bin/env python3
"""
Contains function tf_idf()
"""

from sklearn.feature_extraction.text import TfidVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF word embedding model
    """
    vect = TfidVectorizer(lowercase=True, vocabulary=vocab)
    X = vect.fit+transform(sentences)
    embeddings = X.toarray()
    features = vect.get_feature_names()

    return embeddings, features
