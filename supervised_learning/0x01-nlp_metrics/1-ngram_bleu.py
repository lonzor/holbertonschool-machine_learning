#!/usr/bin/env python3
"""
Contains the following functions:
evaluate_ngram()
ngram_blue()
"""
import numpy as np


def ngram(sentence, n):
    """
    Gets ngrams from sentence
    """
    gram_lst = []

    for word in range(len(sentence) - n + 1):
        gram_lst2 = []
        for gram in range(n):
            gram_lst2.append(sentence[word+gram])
        gram_lst.append(gram_lst2)
    
    return gram_lst


def ngram_bleu(references, sentence, n):
    """
    Gets bleu score
    """
    cal_ngrams = ngram(sentence, n)
    len_grams = len(cal_ngrams)
    sent_len = len(sentence)
    brev = 1
    total = 0

    min_ref = min([len(r) for r in references])
    if sent_len <= min_ref:
        brev = np.exp(1 - min_ref / sent_len)
    while len(cal_ngrams) > 0:
        g = cal_ngrams[0]
        gram_count = cal_ngrams.count(g)
        [cal_ngrams.pop(cal_ngrams.index(g)) for i in range(gram_count)]

        max_r = max([ngram(r, n).count(g) for r in references])
        if gram_count <= max_r:
            total = total + gram_count
        else:
            total = total + max_r

    bleu_score = brev * (total / len_grams)

    return bleu_score
