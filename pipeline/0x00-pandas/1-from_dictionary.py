#!/usr/bin/env python3
"""
Contains function tbat creates a pandas DF from a dictionary
"""
import pandas as pd


diction = {"First": [0.0, 0.5, 1.0, 1.5],
           "Second": ['one', "two", "three", "four"]}

rows = ["A", "B", "C", "D"]

df = pd.DataFrame)diction, index=rows)
print(df)
