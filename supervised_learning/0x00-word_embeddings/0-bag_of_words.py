#!/usr/bin/env python3
"""
contains function bag_of_words()
"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    Creates a matrix using bag of words embedding
    """
    vec_count = CountVectorizer(vocabulary=vocab)
    X = vec_count.fit_transform(sentences)
    features = vec_count.get_feature_names()
    embeddings = X.toarray()

    return embeddings, features
