#!/usr/bin/env python3
"""
Contains function insert_school()
"""
from pymongo import MongoClient


def insert_school(mongo_collection, **kwargs):
    """
    Inserts a new document
    """
    inserted_doc = mongo_collection.insert(kwargs)
    return inserted_doc
