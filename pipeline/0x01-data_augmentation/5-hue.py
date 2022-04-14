#!/usr/bin/env python3
"""
Contains function change_hue()
"""
from tensorflow.image import adjust_hue


def change_hue(iomage, delta):
    """
    Changes hue
    """
    hue_changed = adjust_hue(image, delta)
    return hue_changed
