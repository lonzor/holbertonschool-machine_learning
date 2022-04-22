#!/usr/bin/env python3
"""
Contains function from_numpy(array):
"""
import pandas as pd


def from_numpy(array):
    """
    Creates a pandas dataframe from numpy array
    """
    alpha_list = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    columns = array.shape[1]
    p_data_frame = pd.DataFrame(array, columns=alpha_list[:columns])
    return p_data_frame
