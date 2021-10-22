#!/usr/bin/env python3
"""
contains class FaceAlign
"""
import dlib
import numpy as np


class FaceAlign:
    """
    contains defs that align a face when attempting to detect
    """
    def __init__(self, shape_predictor_path):
        """
        the constructor for FaceAlign
        """
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)
