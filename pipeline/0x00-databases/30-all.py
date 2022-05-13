#!/usr/bin/env python3
"""
Contains function list_all()
"""


def list_all(mongo_collection):
    """
    lists all documents in collection
    """
    docs = []
    collect = mongo_collection.find()
    for i in collect:
        docs.append(i)
    return docs
