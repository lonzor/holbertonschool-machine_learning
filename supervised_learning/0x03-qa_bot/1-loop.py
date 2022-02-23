#!/usr/bin/env python3
"""
contains function question_loop()
"""


def question_loop():
    """
    the loop needed to keep the questions and answers flowing.
    """
    the_door = ["exit", "quit", "goodbye", "bye"]
    empty_text = ""

    while True:
        empty_text = input("Q: ")
        if empty_text.lower() in the_door:
            print("A: Goodbye")
            break
        print("A: ")

if __name__ == "__main__":
    question_loop()
