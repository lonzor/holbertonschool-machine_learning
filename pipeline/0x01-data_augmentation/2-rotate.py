#!/usr/bin/env python3
"""
Contains function rotate_image()
"""
import tensorflow as tf


def rotate_image(image):
    """
    rotates image counter clockwise
    """
    rotated_img = tf.image.rot90(image)
    return rotated_img
