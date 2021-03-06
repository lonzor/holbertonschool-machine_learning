#!/usr/bin/env python3
"""
trains a model using mini-batch gradient descent
uses early stopping and learning rate decay
saves the best iteration of the model
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """
    network is the model to train
    data is a numpy.ndarray of shape (m, nx)
    - containing the input data.
    labels is a one-hot numpy.ndarray of shape (m, classes)
    - containing the labels of data.
    batch_size is the size of the batch used for
    - mini-batch gradient descent.
    epochs is the number of passes through data for
    - mini-batch gradient descent.
    verbose is a boolean that determines if output should
    - be printed during training.
    validation_data is the data used to validate the model
    early_stopping is a boolean
    patience is the patience used for early stopping
    learning_rate_decay is a boolean
    alpha is the initial learning rate
    decay_rate is the decay rate
    Returns history object
    """
    def rate_decay(epoch):
        """
        calculates the learning rate decay for each epoch
        """
        return alpha / (1 + decay_rate * epoch)
    calls = []
    if validation_data is not None:
        if early_stopping:
            calls.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience,
                                                   verbose=verbose))
        if learning_rate_decay:
            calls.append(K.callbacks.LearningRateScheduler(rate_decay))
        if save_best:
            save = K.callbacks.ModelCheckpoint(filepath=filepath,
                                               verbose=verbose,
                                               save_best_only=True)
            calls.append(save)

    hist_obj = network.fit(x=data, y=labels, batch_size=batch_size,
                           epochs=epochs, verbose=verbose, shuffle=shuffle,
                           validation_data=validation_data,
                           callbacks=calls)
    return hist_obj
