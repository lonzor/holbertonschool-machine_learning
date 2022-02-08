#!/usr/bin/env python3
"""
Contains function tf_idf()
"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding model
    """
    vecto = TfidfVectorizer(lowercase=True, vocabulary=vocab)
    X = vecto.fit_transform(sentences)
    embeddings = X.toarray()
    features = vecto.get_feature_names()

    return embeddings, features
