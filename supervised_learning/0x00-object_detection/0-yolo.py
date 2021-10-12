#!/usr/bin/env python3
"""
contains class Yolo
performs object detection
"""
import tensorflow.keras as K


class Yolo:
    """
    the class that will contain main function
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        constructor for the class
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            read_text = f.read()
            split_text = read_text.split('\n')
            if len(split_text[-1]) == 0:
                split_text = split_text[:-1]
        self.class_names = split_text
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
