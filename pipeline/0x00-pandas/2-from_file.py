#!/usr/bin/env python3
"""
contains function from_file(filename, delimiter)
"""
import pandas as pd


def from_file(filename, delimiter):
    """
    loads data from file
    """
    loaded = pd.read_csv(file_name, sep=delimiter)

    return loaded
