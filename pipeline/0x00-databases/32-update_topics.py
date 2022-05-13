#!/usr/bin/env python3
"""
Contains function update_topics()
"""
from pymongo import MongoClient


def update_topics(mongo_collection, name, topics):
    """
    Changes topics of a document based on name
    """
    mongo_collection.update_many({'name': name},
                                 {'Sset': {'name': name, 'topics': topics}})
