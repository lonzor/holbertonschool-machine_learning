#!/usr/bin/env python3
"""
Contains function shear_image(image, intensity)
"""
import tensorflow as tf


def shear_image(image, intensity):
    """
    shears image
    """
    imag_to_arr = tf.keras.preprocessing.image.img_to_array(image)
    sheared_arr = tf.keras.preprocessing.image.random_shear(imag_to_arr,
                                                            intensity)
    sheared_img = tf.keras.preprocessing.image.array_to_img(shear_nparray)
    return sheared_img
