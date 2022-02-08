#!/usr/bin/env python3
"""
Contains fuction fasttext_model()
"""
from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """
    Creates and trains a fasttext embedding model.
    """
    if cbow is True:
        sg = 0
    else:
        sg = 1

    model = FastText(size=size, window=window, min_count=min_count,
                     workers=workers, sg=sg, negative=negative, seed=seed)

    model.build_vocab(sentences=sentences, update=True)
    model.train(sentences=sentences, total_examples=model.corpus_count,
                epochs=iterations)

    return model
