#!/usr/bin/env python3
"""
Contains function change_brightness(image, max_delta)
"""
from tensorflow.image import random_brightness


def change_brightness(image, max_delta):
    """
    Changes brightness of an image randomly
    """
    b_changed = random_brightness(image, max_delta)
    return b_changed
