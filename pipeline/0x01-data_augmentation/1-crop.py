#!/usr/bin/env python3
"""
Contains function crop_image(image,size)
"""
import tensorflow as tf


def crop_image(image, size):
    """
    Crops an image randomly
    """
    cropped_img = tf.image.random_crop(image, size)
    return cropped_img
