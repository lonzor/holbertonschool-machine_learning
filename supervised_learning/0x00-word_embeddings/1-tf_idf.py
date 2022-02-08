#!/usr/bin/env python3
"""
Contains function tf_idf()
"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding model
    """
    vect = TfidfVectorizer(lowercase=True, vocabulary=vocab)
    X = vect.fit_transform(sentences)
    embeddings = X.toarray()
    features = vect.get_feature_names()

    return embeddings, features
    