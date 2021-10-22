#!/usr/bin/env python3
"""
this program will detect and classify faces
"""
import glob
import numpy as np
import cv2
import os
import csv


def load_images(images_path, as_array=True):
    """
    loads images needed to process
    """
    data = []
    file_names = []
    img_paths = glob.glob(images_path + "/*", recursive=False)
    img_paths.sort()

    for img_name in img_paths:
        name = img_name.split('/')[-1]
        file_names.append(name)

    for img_name in img_paths:
        img_read = cv2.imread(img_name)
        img_color = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)
        data.append(img_color)

    if as_array is True:
        data = np.array(data)
    return data, file_names


def load_csv_file(csv_path, params={}):
    """
    loads data from CSV file
    """
    path_to_file = []

    with open(csv_path, 'r') as f:
        read_file = csv.reader(f, params)
        for i in read_file:
            path_to_file.append(i)
    return path_to_file
