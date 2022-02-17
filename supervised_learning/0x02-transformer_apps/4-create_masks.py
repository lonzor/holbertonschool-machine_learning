#!/usr/bin/env python3
"""
Contains function create_masks()
"""
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """
    creates all masks for training/validation
    """
    e_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    e_mask = e_mask[:, tf.newaxis, tf.newaxis, :]
    d_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    d_mask = d_mask[:, tf.newaxis, tf.newaxis, :]

    size = target.shape[1]
    look_mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    t_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    t_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]

    combined = tf.maximum(look_mask, t_mask)

    return e_mask, combined, d_mask
