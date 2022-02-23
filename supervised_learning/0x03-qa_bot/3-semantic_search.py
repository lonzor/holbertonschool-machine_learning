#!/usr/bin/env python3
"""
Contains function semantic_search()
"""
import numpy as np
import os
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on text
    """
    refers = [sentence]

    for i in os.listdir(corpus_path):
        if ".md" in i:
            path = corpus_path + "/" + i
            with open(path, 'r', encoding='utf-8') as f:
                refers.append(f.read())
    mod = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    loader = hub.load(mod)
    embed_refs = embed(refers)
    corr = np.inner(embed_refs, embed_refs)
    index = np.argmax(corr[0, 1:]) + 1

    return refers[index]
