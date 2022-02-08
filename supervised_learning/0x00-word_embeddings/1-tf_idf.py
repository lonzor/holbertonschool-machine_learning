#!/usr/bin/env python3
"""
Contains function tf_idf()
"""
from sklearn.feature_extraction.text import TfidVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates TF-IDF word embedding
    """
    vec = TfidfVectorizer(vocabulary=vocab)
    X = vec.fit_transform(sentences)
    features = vec.get_feaature_names()
    embeddings = X.toarray()

    return embeddings, features
