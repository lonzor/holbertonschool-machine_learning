#!/usr/bin/env python3
"""
Contains the following functions:
ngram()
ngram_bleu()
cumulative_bleu()
"""
import numpy as np


def cumulative_bleu(references, sentence, n):
    """
    Gets cumulative bleu score
    """
    grams_len = len(sentence)
    brev = 1

    ref_mini = min(len(r) for r in references)
    if grams_len <= ref_mini:
        brev = np.exp(1 - ref_mini / grams_len)

    ngram_scores = [ngram_bleu(references, sentence, j) for j in range(1, n + 1)]
    cumulative_score = brev * np.exp(np.log(ngram_scores).sum() / n)
    return cumulative_score


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
    Gets cumulative bleu score
    """
    cal_ngrams = ngram(sentence, n)
    len_grams = len(cal_ngrams)
    total = 0

    while len(cal_ngrams) > 0:
        g = cal_ngrams[0]
        gram_count = cal_ngrams.count(g)
        [cal_ngrams.pop(cal_ngrams.index(g)) for i in range(gram_count)]

        max_r = max([ngram(r, n).count(g) for r in references])
        if gram_count <= max_r:
            total = total + gram_count
        else:
            total = total + max_r

    bleu_score = (total / len_grams)

    return bleu_score
