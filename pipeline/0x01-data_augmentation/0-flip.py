#!/usr/bin/env python3
"""
Contains function tensorflow(image)
"""
import tensorflow as tf


def flip_image(image):
    """
    Flips image horizontally
    """
    flipped_img = tf.image.flip_letf_right(image)
    return flipped_img
