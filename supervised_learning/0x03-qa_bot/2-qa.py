#!/usr/bin/env python3
"""
contains function question_answer()
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


loop = __import__('0-qa').question_answer


def answer_loop(reference):
    """
    Contains the code to produce the command loop for inputs.
    """
    the_door = ["exit", "quit", "goodbye", "bye"]
    empty_text = ""

    while True:
        empty_text = input("Q: ")
        if empty_text.lower() in the_door:
            print("A: Goodbye")
            break
        ans_text = loop(empty_text, reference)
        if ans_text == '':
            print("A: Sorry, I do not understand your question.")
            continue
        print("A: {}".format(answer))
