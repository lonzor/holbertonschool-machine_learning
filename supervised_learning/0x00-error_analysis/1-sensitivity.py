#!/usr/bin/env python3
"""
Contains function sensitivity
"""
import numpy as np


def sensitivity(confusion):
    """
    calculates the sensitivity for each class in a confusion matrix
    confusion is a confusion numpy.ndarray of shape (classes, classes)
    - where row indices represent the correct labels
    - and column indices represent the predicted labels
    classes is the number of classes
    Returns: a numpy.ndarray of shape (classes,)
    - containing the sensitivity of each class
    """
    pos = np.sum(confusion, axis=1)
    true_pos = np.diagonal(confusion)
    sense = true_pos / pos
    return np.array(sense)
