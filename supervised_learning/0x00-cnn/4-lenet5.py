#!/usr/bin/env python3
"""
Contains function lenet5
"""
import tensorflow as tf


def lenet5(x, y):
    """
    builds a modified version of the LeNet-5 architecture using tensorflow
    """
    init = tf.contrib.layers.variance_scaling_initializer()
    con_1 = tf.layers.Conv2D(6, kernel_size=(5, 5), padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=init)(x)
    pool_1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(con_1)
    con_2 = tf.layers.Conv2D(filters=16, kernel_size=5, padding='valid',
                             activation=tf.nn.relu,
                             kernel_initializer=init)(pool_1)
    pool_2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(con_2)

    flat = tf.layers.Flatten()(pool_2)
    c_layer1 = tf.layers.Dense(units=120, kernel_initializer=init,
                               activation=tf.nn.relu)(flat)
    c_layer2 = tf.layers.Dense(units=84, kernel_initializer=init,
                               activation=tf.nn.relu)(c_layer1)
    c_layer3 = tf.layers.Dense(units=10, kernel_initializer=init)(c_layer2)
    pred = c_layer3

    loss = tf.losses.softmax_cross_entropy(y, c_layer3)
    adam_train = tf.train.AdamOptimizer().minimize(loss)
    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    s_max = tf.nn.softmax(pred)
    return s_max, adam_train, loss, acc
