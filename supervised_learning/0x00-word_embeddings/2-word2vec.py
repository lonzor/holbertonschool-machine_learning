#!/usr/bin/env python3
"""
Contains function word2vec()
"""

from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5,
             cbow=True, iterations=5, seed=0, workers=1):
    """
    Creates/trains a word2vec model
    """
    if cbow is True:
        s_gram = 0
    else:
        s_gram = 1

    model = Word2Vec(sentences=sentences, sg=s_gram, negative=negative,
                     window=window, min_count=min_count,
                     workers=workers, seed=seed, size=size)

    model.train(sentences, epochs=iterations,
                total_examples=model.corpus_count)
    return model
