#!/usr/bin/env python3
"""
Contains function uni_bleu()
"""
import numpy as np


def uni_bleu(references, sentence):
    """
    calculates the unigram BLEU score for a sentence
    Order of steps
    1 - count
    2 - clip
    3 - precision
    4 - penalty
    """
    sen_list = list(set(sentence))
    uni_dict = {}

    for i in references:
        for j in i:
            if j in sen_list:
                if j not in uni_dict.keys():
                    uni_dict[j] = i.count(j)
                else:
                    new_count = i.count(j)
                    dict2 = uni_dict[j]
                    uni_dict[j] = max(new_count, dict2)

    len_sent = len(sentence)
    list_ref = []
    for i in references:
        ref_len = len(i)
        list_ref.append(((abs(ref_len - len_sent)), ref_len))

    sorted_ref_len = sorted(list_ref, key=lambda x: x[0])
    sorted_ref_len = sorted_ref_len[0][1]

    if len_sent > sorted_ref_len:
        x = 1
    else:
        x = np.exp(1 - (float(sorted_ref_len) / len_sent))

    b_score = x * np.exp(np.log(sum(uni_dict.values()) / len_sent))

    return b_score