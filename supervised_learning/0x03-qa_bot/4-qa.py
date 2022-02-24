#!/usr/bin/env python3
"""
Contains the function qa_bot()
"""


loop = __import__('0-qa').question_answer
search = __import__('3-semantic_search').semantic_search


def qa_bot(corpus_path):
    """
    Creates the loop with the ability to search the corpus for matching
    input text.
    """
    input_q = ''
    the_door = ["exit", "quit", "goodbye", "bye"]


    while True:
        input_q = input("Q: ")
        if input_q.lower() in the_door:
            print("A: Goodbye")
            break
        answer_text = semantic_search(corpus_path, input_q)
        response = qa(input_q, answer_text)
        if response == '':
            print("A: Sorry, I do not understand your question.")
            continue
        print("A: {}".format(response))
