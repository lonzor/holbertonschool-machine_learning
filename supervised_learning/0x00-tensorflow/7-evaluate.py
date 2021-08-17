#!/usr/bin/env python3
"""contains the evaluate method"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    X is a numpy.ndarray containing the input data to evaluate
    Y is a numpy.ndarray containing the one-hot labels for X
    save_path is the location to load the model from
    """
    with tf.Session() as sess:
        saved = tf.train.import_meta_graph(save_path + '.meta')
        saved.restore(sess, save_path)

        y = tf.get_collection("y")[0]
        x = tf.get_collection("x")[0]
        acc = tf.get_collection("acc")[0]
        y_pred = tf.get_collection("y_pred")[0]
        loss = tf.get_collection("loss")[0]

        pred = sess.run(y_pred, feed_dict={x: X, y: Y})
        e_acc = sess.run(acc, feed_dict={x: X, y: Y})
        e_loss = sess.run(loss, feed_dict={x: X, y: Y})
        return pred, e_acc, e_loss
