#!/usr/bin/env python3
"""Contains method to calculate accuracey with tensorflow"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network’s predictions
    """
    y = tf.argmax(y, 1)
    pred = tf.argmax(y_pred, 1)
    eq = tf.equal(pred, y)
    acc = tf.reduce_mean(tf.cast(eq, tf.float32))
    return acc
