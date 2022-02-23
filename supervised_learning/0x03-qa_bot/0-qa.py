#!/usr/bin/env python3
"""
contains function question_answer()
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question
    """
    tokize = BertTokenizer.from_pretrained
    tokizer = tokize('bert-larg-uncased-whole-word-masking-finetuned-squad')
    mdl = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')
    quest = tokizer.tokenize(questtion)
    refer = tokizer.tokenize(reference)
    tkns = ['[CLS]'] + quest + ['[SEP]'] + refer + ['[SEP]']
    input_ids = tokizer.convert_tokens_to_ids(tkns)
    mask = [1] * len(input_ids)
    type_ids = [0] * (1 + len(quest) + 1) + [1] * (len(refer) + 1)

    input_ids, mask, type_ids = map(lambda t: tf.expand_dims(
        tf.convert_to_tensor(t, dtype=tf.int32), 0),
        (input_ids, mask, type_ids))

    outs = model([input_ids, mask, type_ids])
    start = tf.argmax(outs[0][0][1:]) + 1
    end = tf.argmax(outs[1][0][1:]) + 1
    a_toks = tkns[start: end + 1]
    answer = tokizer.convert_tokens_to_string(a_toks)

    return answer
