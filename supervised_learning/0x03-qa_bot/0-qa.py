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
    text = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    tokenizer = BertTokenizer.from_pretrained(text)
    model = hub.load("https://tfhub.dv/see--/bert-uncased-tf2-qa/1")
    quest = tokenizer.tokenize(questtion)
    refer = tokenizer.tokenize(reference)
    tkns = ['[CLS]'] + quest + ['[SEP]'] + refer + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tkns)
    mask = [1] * len(input_ids)
    type_ids = [0] * (1 + len(quest) + 1) + [1] * (len(refer) + 1)

    input_ids, mask, type_ids = map(lambda t: tf.expand_dims(
        tf.convert_to_tensor(t, dtype=tf.int32), 0),
        (input_ids, mask, type_ids))

    outs = model([input_ids, mask, type_ids])
    start = tf.argmax(outs[0][0][1:]) + 1
    end = tf.argmax(outs[1][0][1:]) + 1
    a_toks = tkns[start: end + 1]
    answer = tokenizer.convert_tokens_to_string(a_toks)

    if answer is None or answer == '':
        answer = None

    return answer
