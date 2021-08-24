#!/usr/bin/env python3
"""
Trains a loaded neural network model using mini-batch gradient descent
"""
import tensorflow as tf


shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    X_train a numpy.ndarray of shape (m, 784) containing the training data
    m is the number of data points
    784 is the number of input features
    Y_train is a one-hot numpy.ndarray of shape (m, 10)
    10 is the number of classes the model should classify
    X_valid a numpy.ndarray of shape (m, 784) containing the validation data
    Y_valid is a one-hot numpy.ndarray of shape (m, 10)
    batch_size is the number of data points in a batch
    epochs is the number of times the training should pass through
    - the whole dataset
    load_path is the path from which to load the model
    save_path is the path to where the model should be saved after training
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        for ep in range(epochs + 1):
            tcost = loss.eval({x: X_train, y: Y_train})
            tacc = accuracy.eval({x: X_train, y: Y_train})
            vcost = loss.eval({x: X_valid, y: Y_valid})
            vacc = accuracy.eval({x: X_valid, y: Y_valid})
            print("After {} epochs:".format(ep))
            print("\tTraining Cost: {}".format(tcost))
            print("\tTraining Accuracy: {}".format(tacc))
            print("\tValidation Cost: {}".format(vcost))
            print("\tValidation Accuracy: {}".format(vacc))

            if ep < epochs:
                Xshuf, Yshuf = shuffle_data(X_train, Y_train)
                for i in range(0, X_train.shape[0], batch_size):
                    feed_dict = {x: Xshuf[i:i + batch_size],
                                 y: Yshuf[i:i + batch_size]}
                    sess.run(train_op, feed_dict)
                    if not ((i // batch_size + 1) % 100):
                        i_loss = loss.eval(feed_dict)
                        i_acc = accuracy.eval(feed_dict)
                        print("\tStep {}:".format(i//batch_size + 1))
                        print("\ttCost {}:".format(i_loss))
                        print("\ttAccuracy {}:".format(i_acc))
        return saver.save(sess, save_path)
